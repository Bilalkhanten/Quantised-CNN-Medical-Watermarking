"""Sanity check: can the encoder/decoder pair learn to carry 511 bits at all? (synthetic data, no attacks, CPU-friendly)"""
import sys; sys.path.insert(0, '.')
import torch, torch.nn.functional as F, time
import data as D
from system import *; from payload import PayloadFactory; from common import *
seed_all(0); ds = D.SyntheticSet(32); fac = PayloadFactory(seed=0)
s = System(dict()); opt = torch.optim.AdamW(list(s.enc.parameters())+list(s.dec.parameters()), 1e-3)
bank = [fac.batch(8) for _ in range(40)]                       # reuse a bank of valid payloads for speed
t0 = time.time()
for it in range(400):
    idx = torch.randint(0, len(ds), (8,)); img = torch.stack([ds[i][0] for i in idx]); m, rk, _ = bank[it % 40]; m, rk = torch.from_numpy(m), torch.from_numpy(rk)
    iw = s.embed(img, m, rk); lg = s.extract(iw)
    L = F.binary_cross_entropy_with_logits(lg, m) + 20*F.mse_loss(iw, img)
    opt.zero_grad(); L.backward(); opt.step()
    if it % 50 == 0 or it == 399:
        ber = ((lg > 0).float() != m).float().mean().item(); print(f'it {it:3d} L_msg-ish {L.item():.3f} BER {ber*100:.1f}% PSNR {psnr(img, iw).mean().item():.1f} dB  ({time.time()-t0:.0f}s)', flush=True)
