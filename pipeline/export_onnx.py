"""Export the pruned encoder (and decoder) to ONNX and quantise to *real* INT8 (static, QDQ) for on-device timing.  The PyTorch 'int8sim' model uses fake-quantisation,
which still computes in FP32; only an exported INT8 graph run by an INT8 runtime (ONNX Runtime / XNNPACK / TFLite) gives honest INT8 latency and energy.
  python export_onnx.py --run runs/full_s0 --calib /data/mias --out runs/full_s0/onnx
Produces enc_fp32.onnx, enc_int8.onnx, dec_fp32.onnx, dec_int8.onnx and onnx_report.json (max |FP32-ONNX| deviation, model sizes)."""
import argparse, json, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from system import *; from models import Encoder, Decoder, haar_dwt
import data as D, payload as P

class EncW(nn.Module):
    def __init__(s, e): super().__init__(); s.e = e
    def forward(s, img, m, rk): return s.e(img, m, rk)
class DecW(nn.Module):
    """same computation as Decoder.forward for 256x256 inputs; adaptive_avg_pool2d(32x32 -> 4x4) is written as the equivalent fixed 8x8 average pool (or 32x32 for global pooling) so it exports"""
    def __init__(s, d): super().__init__(); s.d = d
    def forward(s, img):
        d = s.d; x = d.stn(img)[0] if d.stn is not None else img
        x = d.conv(d.blocks(d.stem(haar_dwt(x)))); x = F.avg_pool2d(x, 32 if d.global_pool else 8); return d.fc(x.flatten(1))
class Reader(CalibrationDataReader):
    def __init__(s, feeds): s.it = iter(feeds)
    def get_next(s): return next(s.it, None)

def load_fp32(run):
    meta = json.load(open(os.path.join(run, 'meta.json'))); sd = torch.load(os.path.join(run, 'stage4_pruned_fp32.pt'), map_location='cpu')
    C = sd['enc.stem.0.weight'].shape[0]; sysm = System(meta['cfg']); sysm.enc = Encoder(C=C, alpha=float(sd.get('enc.alpha', 1.0)) if 'enc.alpha' in sd else 1.0)
    sysm.load_state_dict({k: v for k, v in sd.items()}, strict=False); return sysm.eval(), meta
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--calib'); ap.add_argument('--n_calib', type=int, default=64); ap.add_argument('--out'); ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--alpha', type=float, default=None, help='embedding strength used at deployment (the calibrated 40 dB value from evaluate.py)'); a = ap.parse_args()
    out = a.out or os.path.join(a.run, 'onnx'); os.makedirs(out, exist_ok=True); sysm, meta = load_fp32(a.run)
    if a.alpha: sysm.enc.alpha = a.alpha
    ds = D.SyntheticSet(a.n_calib, seed=9) if a.synthetic else D.ImageSet(D.patient_split(D.mias_records(a.calib))['train'][:a.n_calib]); fac = P.PayloadFactory(meta['variant'], seed=11)
    img = ds[0][0][None]; m, rk, _ = fac.batch(1); m, rk = torch.from_numpy(m), torch.from_numpy(rk); rep = {}
    torch.onnx.export(EncW(sysm.enc), (img, m, rk), f'{out}/enc_fp32.onnx', input_names=['img', 'm', 'rk'], output_names=['iw'], opset_version=20, dynamo=False)
    torch.onnx.export(DecW(sysm.dec), (img,), f'{out}/dec_fp32.onnx', input_names=['img'], output_names=['logits'], opset_version=20, dynamo=False)
    for name, feeds in (('enc', None), ('dec', None)):
        fe = []
        for i in range(a.n_calib):
            x = ds[i % len(ds)][0][None].numpy(); mm, rr, _ = fac.batch(1); fe.append(dict(img=x, m=mm, rk=rr) if name == 'enc' else dict(img=x))
        quantize_static(f'{out}/{name}_fp32.onnx', f'{out}/{name}_int8.onnx', Reader(fe), quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True)
        rep[f'{name}_fp32_bytes'] = os.path.getsize(f'{out}/{name}_fp32.onnx'); rep[f'{name}_int8_bytes'] = os.path.getsize(f'{out}/{name}_int8.onnx')
    sysm.eval()                       # torch.onnx.export leaves submodules in train mode; the reference must use BN running statistics
    s = ort.InferenceSession(f'{out}/enc_fp32.onnx'); y = s.run(None, dict(img=img.numpy(), m=m.numpy(), rk=rk.numpy()))[0]
    with torch.no_grad(): rep['enc_fp32_max_abs_diff_vs_torch'] = float(np.abs(y-sysm.enc(img, m, rk).numpy()).max())
    s8 = ort.InferenceSession(f'{out}/enc_int8.onnx'); y8 = s8.run(None, dict(img=img.numpy(), m=m.numpy(), rk=rk.numpy()))[0]; rep['enc_int8_mean_abs_diff_vs_fp32'] = float(np.abs(y8-y).mean())
    sd_ = ort.InferenceSession(f'{out}/dec_fp32.onnx'); ld = sd_.run(None, dict(img=img.numpy()))[0]
    with torch.no_grad(): rep['dec_fp32_max_abs_diff_vs_torch'] = float(np.abs(ld-sysm.dec(img)[0].numpy()).max())
    json.dump(rep, open(f'{out}/onnx_report.json', 'w'), indent=1); print(rep)
if __name__ == '__main__': main()
