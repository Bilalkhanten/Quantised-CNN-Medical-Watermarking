"""Per-stage latency, peak memory and (optionally) energy on the target device (Table 13).  Run ON THE Raspberry Pi 4/5 (or any device).
  python measure_cost.py --run runs/full_s0 --n 200 [--power_csv trace.csv --idle_w 2.7]
Energy: with a USB power meter logging `t_seconds,watts` to a CSV, the script marks wall-clock stage boundaries (time.time()) and integrates (P - P_idle) over each
stage, E_stage = integral (P(t) - P_idle) dt  (Eq. 26).  Run the stage in a loop of --repeat executions so the integral is well above the meter's resolution; the script divides back.
Without --power_csv only latency / memory are reported (no energy is invented)."""
import argparse, json, time, resource, numpy as np, torch, os
import payload as P, evaluate as E, attacks as A
from models import haar_dwt, haar_idwt
from common import *

def p95(v): return float(np.percentile(v, 95))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--n', type=int, default=200); ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--power_csv'); ap.add_argument('--idle_w', type=float, default=0.0); ap.add_argument('--threads', type=int, default=4); ap.add_argument('--out')
    ap.add_argument('--onnx_dir', help='directory written by export_onnx.py: times the exported FP32 and INT8 graphs with ONNX Runtime')
    a = ap.parse_args(); torch.set_num_threads(a.threads); sysm, meta = E.load_run(a.run, torch.device('cpu')); sysm.eval(); fac = P.PayloadFactory(meta['variant'], seed=5)
    img = torch.rand(1, 1, 256, 256); marks = {}; lat = {k: [] for k in ('payload_encode', 'embed', 'extract', 'verify', 'localise')}
    def stage(name, fn):
        t0 = time.time(); ts = []
        for _ in range(a.n):
            t = time.perf_counter(); out = fn(); ts.append(time.perf_counter()-t)
        marks[name] = (t0, time.time()); lat[name] = ts; return out
    with torch.no_grad():
        stage('payload_encode', lambda: fac.batch(1)); m, rk, aux = fac.batch(1); mt, rkt = torch.from_numpy(m), torch.from_numpy(rk)
        for _ in range(5): sysm.embed(img, mt, rkt)                                                   # warm-up
        stage('dwt_idwt', lambda: haar_idwt(haar_dwt(img))); iw = stage('embed', lambda: sysm.embed(img, mt, rkt)); stage('extract', lambda: sysm.extract(iw))
        bits = (sysm.extract(iw) > 0).numpy().astype('uint8')[0]; key, md, iv = aux[0]; stage('verify', lambda: P.verify(bits, key, md, meta['variant'])); stage('localise', lambda: sysm.localise(iw, rkt))
    if a.onnx_dir:
        import onnxruntime as ort; so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
        feeds = dict(enc=dict(img=img.numpy(), m=mt.numpy(), rk=rkt.numpy()), dec=dict(img=iw.numpy()))
        for name in ('enc_fp32', 'enc_int8', 'dec_fp32', 'dec_int8'):
            sess = ort.InferenceSession(os.path.join(a.onnx_dir, name+'.onnx'), so, providers=['CPUExecutionProvider']); f = feeds[name[:3]]
            for _ in range(5): sess.run(None, f)
            stage('onnx_'+name, lambda: sess.run(None, f))
    res = {k: dict(median_ms=1e3*float(np.median(v)), p95_ms=1e3*p95(v), mean_ms=1e3*float(np.mean(v))) for k, v in lat.items()}
    res['sender_total_ms'] = res['payload_encode']['median_ms']+res['embed']['median_ms']; res['sender_total_int8_ms'] = res['payload_encode']['median_ms']+res['onnx_enc_int8']['median_ms'] if a.onnx_dir else None; res['receiver_total_ms'] = res['extract']['median_ms']+res['verify']['median_ms']
    res['receiver_with_localiser_ms'] = res['receiver_total_ms']+res['localise']['median_ms']; res['peak_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    res['device'] = dict(machine=os.uname().machine, torch_threads=a.threads, torch=torch.__version__)
    if a.power_csv:
        T = np.loadtxt(a.power_csv, delimiter=',', skiprows=1); t, w = T[:, 0], T[:, 1]; en = {}
        for k, (t0, t1) in marks.items():
            sel = (t >= t0) & (t <= t1)
            if sel.sum() >= 2: en[k] = 1e3*float(np.trapz(np.clip(w[sel]-a.idle_w, 0, None), t[sel]))/(a.n*a.repeat)         # mJ per execution
        res['energy_mJ_per_exec'] = en
    print(json.dumps(res, indent=1)); json.dump(res, open(a.out or os.path.join(a.run, 'cost.json'), 'w'), indent=1)
if __name__ == '__main__': main()
