"""Baselines for Table 10 / 11, evaluated with the *same* payload (BCH-coded 511 bits), attacks, 8-bit rounding and 40 dB operating point as the proposed system.
  dwtsvd : classical blind DWT-SVD, QIM on the largest singular value of 4x4 LL blocks (implemented here, no training)
  hidden : HiDDeN-style CNN (message map concatenated to image, plain CNN decoder, no DWT / reference pattern / STN), trained here with the same attack layer
  redmark, mbrs : NOT re-implemented.  Use the authors' official code; wrap it with the embed()/extract() interface below and call run_eval().
Usage:  python baselines.py dwtsvd --mias /data/mias --out runs/base_dwtsvd.json
        python baselines.py hidden --mias /data/mias --epochs 60 --out runs/base_hidden.json"""
import argparse, json, os, sys, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
import data as D, payload as P, evaluate as E
from models import haar_dwt, haar_idwt, DSBlock
from attacks import AttackLayer, q8
from common import *

class _Enc:                                    # exposes .alpha so evaluate.calibrate_alpha works on any baseline
    def __init__(s, alpha): s.alpha = alpha

class DWTSVD(nn.Module):
    """alpha = QIM step (in LL-coefficient units, 0..1 image scale).  Embeds the 511 coded bits into the first 511 of the 32x32 grid of 4x4 LL blocks."""
    def __init__(s, alpha=0.05): super().__init__(); s.enc = _Enc(alpha)
    def _blocks(s, ll):                        # B,1,128,128 -> B,1024,4,4
        return ll.unfold(2, 4, 4).unfold(3, 4, 4).reshape(ll.shape[0], -1, 4, 4)
    def _unblocks(s, b): return b.reshape(b.shape[0], 32, 32, 4, 4).permute(0, 1, 3, 2, 4).reshape(b.shape[0], 1, 128, 128)
    @torch.no_grad()
    def embed(s, img, m, rk=None):
        y = haar_dwt(img); b = s._blocks(y[:, :1]); U, S, Vh = torch.linalg.svd(b); d = s.enc.alpha
        bit = torch.zeros_like(S[:, :, 0]); bit[:, :511] = m; s1 = S[:, :, 0]; q = d*torch.round((s1-bit*d/2)/d)+bit*d/2
        S2 = S.clone(); S2[:, :511, 0] = q[:, :511]; b2 = U@torch.diag_embed(S2)@Vh
        y2 = torch.cat([s._unblocks(b2), y[:, 1:]], 1); return q8(haar_idwt(y2).clamp(0, 1))
    @torch.no_grad()
    def extract(s, att):
        b = s._blocks(haar_dwt(att)[:, :1]); S = torch.linalg.svdvals(b)[:, :511, 0]; d = s.enc.alpha
        d0 = (S-d*torch.round(S/d)).abs(); d1 = (S-(d*torch.round((S-d/2)/d)+d/2)).abs(); return d0-d1      # >0 -> bit 1
    def localise(s, img, rk): raise NotImplementedError('classical baseline has no localiser')

class Hidden(nn.Module):
    """HiDDeN-style: encoder = conv stack on [image, message map]; decoder = conv stack + global pool + linear.  Same capacity class as the proposed model, none of its design elements."""
    def __init__(s, C=32, alpha=1.0, n_bits=511):
        super().__init__(); s.enc = _EncNet(C, alpha, n_bits); s.dec = _DecNet(C, n_bits)
    def embed(s, img, m, rk=None): return q8(s.enc(img, m).clamp(0, 1))
    def extract(s, att): return s.dec(att)
class _EncNet(nn.Module):
    def __init__(s, C, alpha, n_bits):
        super().__init__(); s.alpha = alpha; s.msg = nn.Linear(n_bits, 1024)
        s.net = nn.Sequential(nn.Conv2d(1+4, C, 3, 1, 1), nn.BatchNorm2d(C), nn.ReLU(), nn.Conv2d(C, C, 3, 1, 1), nn.BatchNorm2d(C), nn.ReLU(), nn.Conv2d(C, C, 3, 1, 1), nn.BatchNorm2d(C), nn.ReLU(), nn.Conv2d(C, 1, 1))
    def forward(s, img, m):
        mm = F.interpolate(s.msg(m*2-1).view(-1, 4, 16, 16), size=img.shape[-2:], mode='bilinear', align_corners=False)
        return img+s.alpha*s.net(torch.cat([img, mm], 1))
class _DecNet(nn.Module):
    def __init__(s, C, n_bits):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(1, C, 3, 2, 1), nn.BatchNorm2d(C), nn.ReLU(), nn.Conv2d(C, 2*C, 3, 2, 1), nn.BatchNorm2d(2*C), nn.ReLU(), nn.Conv2d(2*C, 4*C, 3, 2, 1), nn.BatchNorm2d(4*C), nn.ReLU(), nn.Conv2d(4*C, 4*C, 3, 2, 1), nn.BatchNorm2d(4*C), nn.ReLU())
        s.fc = nn.Linear(4*C*16*16//16, n_bits) if False else nn.LazyLinear(n_bits)
    def forward(s, x): return s.fc(F.adaptive_avg_pool2d(s.net(x), 4).flatten(1))

def train_hidden(model, loader, fac, dev, epochs, lr=1e-3, l_img=1.0, msg_warmup=2):
    att = AttackLayer(); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4); model.to(dev)
    with torch.no_grad(): model.dec(torch.zeros(1, 1, 256, 256, device=dev))            # materialise the lazy layer
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        model.train(); w = min(1.0, ep/msg_warmup) if msg_warmup else 1.0; s = min(1.0, (ep+1)/max(epochs//3, 1))
        for img, roi in loader:
            img = img.to(dev); m, rk, _ = fac.batch(img.shape[0]); m = torch.from_numpy(m).to(dev); iw = model.embed(img, m)
            a, _ = att(iw, s); L = F.binary_cross_entropy_with_logits(model.extract(a), m)+l_img*w*20*F.mse_loss(iw, img)
            opt.zero_grad(); L.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        print(f'[hidden] epoch {ep} L {L.item():.4f}', flush=True)

def run_eval(model, te, va, dev, variant='cbc', out=None, seed=123):
    """Calibrate to 40 dB on validation, then evaluate the Table-10 + unseen attacks on the test split.  Same code path as evaluate.py."""
    seed_all(seed); model.to(dev).eval(); fv, ft = P.PayloadFactory(variant, seed=777), P.PayloadFactory(variant, seed=888)
    a = E.calibrate_alpha(model, va, fv, dev); pay, fid = E.eval_payload(model, te, dev, ft, E.TABLE10+E.UNSEEN, variant, True)
    r = dict(alpha=float(a), attacks=pay, fidelity=fid, params=dict(enc=sum(p.numel() for p in model.enc.parameters()) if hasattr(model.enc, 'parameters') else 0)); 
    if out: json.dump(r, open(out, 'w'))
    print('PSNR %.2f | exact (10 seen) %.1f%%' % (np.mean(fid['psnr']), 100*np.mean([np.mean(pay[f'{n}|{s}']['exact']) for n, s, _ in E.TABLE10]))); return r

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('method', choices=['dwtsvd', 'hidden']); ap.add_argument('--mias'); ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=16); ap.add_argument('--out', required=True); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--synthetic', type=int, default=0)
    ap.add_argument('--split_seed', type=int, default=0); a = ap.parse_args(); dev = get_device(); seed_all(a.seed)
    if a.synthetic: tr, va, te = (torch.utils.data.DataLoader(D.SyntheticSet(a.synthetic, seed=s), batch_size=8, shuffle=(s == 1)) for s in (1, 2, 3))
    else:
        sp = D.patient_split(D.mias_records(a.mias), seed=a.split_seed); tr = torch.utils.data.DataLoader(D.ImageSet(sp['train'], train=True), batch_size=a.bs, shuffle=True, drop_last=True)
        va, te = E.make_loader(sp['val']), E.make_loader(sp['test'])
    if a.method == 'dwtsvd': model = DWTSVD()
    else: model = Hidden(); train_hidden(model, tr, P.PayloadFactory('cbc', seed=a.seed), dev, a.epochs)
    run_eval(model, te, va, dev, out=a.out)
