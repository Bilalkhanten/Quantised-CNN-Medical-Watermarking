"""Four-stage training of the proposed framework (Section 3.6).  Example:
  python train.py --mias /data/mias --pretrain /data/cbis /data/chestxray14 --out runs/full --seed 0
Ablation flags: --no_attack (i) --no_bch (ii) --no_stn (iii) --no_roi (iv) --spatial_encoder (v) --global_pool (viii) --no_rk (ix) --no_d (xiii)
Compression variants: --ptq (xi: post-training quantisation instead of QAT)  --no_finetune (xii: prune without fine-tuning)  --prune 0.38"""
import argparse, os, json, time, math, copy, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import data as D
from system import *
from attacks import AttackLayer, tamper, q8, jpeg_real
from payload import PayloadFactory
from common import *

def make_loaders(a):
    if a.synthetic:
        tr = D.SyntheticSet(a.synthetic, seed=1); va = D.SyntheticSet(16, seed=2); te = D.SyntheticSet(16, seed=3); pre = D.SyntheticSet(a.synthetic, seed=4)
    else:
        recs = D.mias_records(a.mias); sp = D.patient_split(recs, seed=a.split_seed)
        tr, va, te = [D.ImageSet(sp[k], train=(k == 'train')) for k in ('train', 'val', 'test')]
        pre = ConcatDataset([D.ImageSet(D.folder_records(p, limit=a.pretrain_limit), train=True) for p in a.pretrain]) if a.pretrain else tr
        json.dump({k: [r['name'] for r in v] for k, v in sp.items()}, open(os.path.join(a.out, 'split.json'), 'w'))
    mk = lambda ds, sh: DataLoader(ds, batch_size=a.bs, shuffle=sh, drop_last=sh, num_workers=a.workers)
    return mk(pre, True), mk(tr, True), mk(va, False), mk(te, False)

def lr_at(e, E=100, hi=1e-3, lo=1e-5): return lo+0.5*(hi-lo)*(1+math.cos(math.pi*e/E))   # Eq. (23)

@torch.no_grad()
def val_exact(sysm, loader, fac, dev, n_max=24):
    """model-selection score on the validation split: mean exact-decode rate under JPEG-70 / noise-5 / identity, and mean PSNR"""
    from attacks import jpeg_real, gauss_noise
    import payload as P
    sysm.eval(); ok = tot = 0; ps = []
    for img, roi in loader:
        img = img.to(dev); m, rk, aux = fac.batch(img.shape[0]); mt, rkt = torch.from_numpy(m).to(dev), torch.from_numpy(rk).to(dev)
        iw = sysm.embed(img, mt, rkt); ps.append(psnr(img, iw).mean().item())
        for att in (iw, q8(jpeg_real(iw, 70).clamp(0, 1)), q8(gauss_noise(iw, 5).clamp(0, 1))):
            bits_ = (sysm.extract(att) > 0).cpu().numpy().astype('uint8')
            for b, (key, md, iv) in zip(bits_, aux): ok += P.verify(b, key, md, fac.variant) == 'authentic'; tot += 1
        if tot >= n_max*3: break
    return ok/max(tot, 1), sum(ps)/len(ps)

def step_losses(sysm, img, roi, m, rk, a, strength, attack, with_loc, other):
    iw = sysm.embed(img, m, rk)
    att, _ = attack(iw, strength) if a.attack else (iw, None)
    logits = sysm.extract(att)
    Limg = img_loss(img, iw, roi, 1.0 if a.no_roi else a.omega, a.beta)
    Lmsg = F.binary_cross_entropy_with_logits(logits, m)
    L = a.l_img*getattr(a, '_w', 1.0)*Limg + a.l_msg*Lmsg; Lloc = torch.zeros(())
    if with_loc:
        tmp, mask = tamper(iw.detach(), other)
        benign = torch.rand(img.shape[0], 1, 1, 1, device=img.device) < 0.5          # half of the samples stay untampered -> all-zero label
        x_loc = torch.where(benign, q8(jpeg_real(iw.detach(), float(torch.randint(70, 91, ())))).clamp(0, 1), tmp)
        mask = mask*(~benign).float(); pred = sysm.localise(x_loc, rk)
        mk = to_mask128(mask); Lloc = F.binary_cross_entropy(pred, mk)+dice_loss(pred, mk); L = L+a.l_loc*Lloc
    return L, Limg.detach(), Lmsg.detach(), Lloc.detach()

def run_epochs(sysm, loader, params, a, fac, attack, dev, epochs, lr_fn, stage, curriculum=False, with_loc=False, slim=0.0, log=None, start_epoch=0):
    opt = torch.optim.AdamW(params, lr=lr_fn(0), weight_decay=1e-4)
    for ep in range(epochs):
        for g in opt.param_groups: g['lr'] = lr_fn(ep)
        s = min(1.0, (ep+1)/a.curriculum_epochs) if curriculum else 1.0
        a._w = min(1.0, (start_epoch+ep)/a.msg_warmup) if a.msg_warmup else 1.0        # optional message-first warm-up of the image-loss weight
        sysm.train(); t0 = time.time(); agg = torch.zeros(3)
        for it, (img, roi) in enumerate(loader):
            img, roi = img.to(dev), roi.to(dev); m, rk, _ = fac.batch(img.shape[0]); m, rk = torch.from_numpy(m).to(dev), torch.from_numpy(rk).to(dev)
            L, a1, a2, a3 = step_losses(sysm, img, roi, m, rk, a, s, attack, with_loc, img.flip(0))
            if slim: L = L+slim*slim_penalty(sysm.enc)
            opt.zero_grad(); L.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); agg += torch.tensor([a1, a2, a3])
            if a.probe and it % a.probe == 0: print(f'[probe] stage {stage} ep {start_epoch+ep} it {it} L_msg {a2.item():.4f} (0.693 = chance plateau) L_img {a1.item():.4f} w_img {a._w:.2f}', flush=True)
        agg /= max(len(loader), 1)
        if log: log(dict(stage=stage, epoch=start_epoch+ep, strength=s, lr=lr_fn(ep), L_img=agg[0].item(), L_msg=agg[1].item(), L_loc=agg[2].item(), sec=time.time()-t0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mias'); ap.add_argument('--pretrain', nargs='*', default=[]); ap.add_argument('--pretrain_limit', type=int, default=None)
    ap.add_argument('--out', default='runs/run'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--split_seed', type=int, default=0)
    ap.add_argument('--bs', type=int, default=16); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--e1', type=int, default=20); ap.add_argument('--e2', type=int, default=80); ap.add_argument('--e2_ft', type=int, default=20)
    ap.add_argument('--e3', type=int, default=20); ap.add_argument('--e4_slim', type=int, default=10); ap.add_argument('--e4_ft', type=int, default=10); ap.add_argument('--e4_qat', type=int, default=10)
    ap.add_argument('--curriculum_epochs', type=int, default=30); ap.add_argument('--prune', type=float, default=0.38); ap.add_argument('--l_slim', type=float, default=1e-4)
    ap.add_argument('--l_img', type=float, default=1.0); ap.add_argument('--l_msg', type=float, default=1.0); ap.add_argument('--l_loc', type=float, default=0.5)
    ap.add_argument('--beta', type=float, default=0.1); ap.add_argument('--omega', type=float, default=5.0); ap.add_argument('--variant', default='cbc', choices=['cbc', 'hmac'])
    for f in ('no_attack', 'no_bch', 'no_stn', 'no_roi', 'spatial_encoder', 'global_pool', 'no_rk', 'no_d', 'ptq', 'no_finetune', 'synthetic_flag'): ap.add_argument('--'+f, action='store_true')
    ap.add_argument('--msg_warmup', type=int, default=0, help='epochs over which the image-loss weight ramps 0->1 (0 = off, paper-faithful)'); ap.add_argument('--probe', type=int, default=0, help='print L_msg every N steps to see whether training leaves the 0.693 chance plateau')
    ap.add_argument('--synthetic', type=int, default=0, help='N synthetic images (smoke test only)')
    a = ap.parse_args(); a.attack = not a.no_attack; os.makedirs(a.out, exist_ok=True)
    seed_all(a.seed); dev = get_device(); json.dump(vars(a), open(os.path.join(a.out, 'config.json'), 'w'), indent=1)
    logf = open(os.path.join(a.out, 'train_log.jsonl'), 'a'); log = lambda d: (logf.write(json.dumps(d)+'\n'), logf.flush(), print(d))
    pre, tr, va, te = make_loaders(a)
    fac = PayloadFactory(a.variant, seed=a.seed, coded=not a.no_bch); fac_val = PayloadFactory(a.variant, seed=10_000+a.seed, coded=not a.no_bch)
    cfg = dict(no_stn=a.no_stn, global_pool=a.global_pool, spatial_encoder=a.spatial_encoder, no_rk=a.no_rk, no_d=a.no_d)
    sysm = System(cfg).to(dev); attack = AttackLayer(); E = a.e1+a.e2; gl = lambda e0: (lambda e: lr_at(e0+e, E))
    ed = list(sysm.enc.parameters())+list(sysm.dec.parameters())
    # Stage 1: encoder + decoder, no attacks, on pre-training data
    run_epochs(sysm, pre, ed, a, fac, lambda x, s: (x, None), dev, a.e1, gl(0), 1, log=log)
    # Stage 2: attack curriculum (Eq. 22), then fine-tune on the MIAS training split
    run_epochs(sysm, pre, ed, a, fac, attack, dev, a.e2, gl(a.e1), 2, curriculum=True, log=log, start_epoch=a.e1)
    run_epochs(sysm, tr, ed, a, fac, attack, dev, a.e2_ft, lambda e: 1e-5, '2-finetune', log=log)
    # Stage 3: add the localiser, all networks unfrozen at a low learning rate
    run_epochs(sysm, tr, list(sysm.parameters()), a, fac, attack, dev, a.e3, lambda e: 1e-4, 3, with_loc=True, log=log)
    ex, ps = val_exact(sysm, va, fac_val, dev); log(dict(stage='3-val', exact=ex, psnr=ps)); torch.save(sysm.state_dict(), os.path.join(a.out, 'stage3.pt'))
    # Stage 4: network slimming -> channel pruning -> fine-tune -> QAT (or PTQ for ablation xi)
    run_epochs(sysm, tr, list(sysm.parameters()), a, fac, attack, dev, a.e4_slim, lambda e: 1e-4, '4-slim', with_loc=True, slim=a.l_slim, log=log)
    sysm.enc, red = prune_encoder(sysm.enc, a.prune); sysm.enc.to(dev); log(dict(stage='4-prune', channel_reduction=red, enc_params=count_params(sysm.enc)))
    if not a.no_finetune: run_epochs(sysm, tr, list(sysm.parameters()), a, fac, attack, dev, a.e4_ft, lambda e: 5e-5, '4-finetune', with_loc=True, log=log)
    torch.save(sysm.state_dict(), os.path.join(a.out, 'stage4_pruned_fp32.pt'))
    sysm.enc = to_qat(sysm.enc).to(dev)
    if a.ptq:                                              # calibrate observers only
        sysm.eval()
        with torch.no_grad():
            for img, roi in tr: m, rk, _ = fac.batch(img.shape[0]); sysm.embed(img.to(dev), torch.from_numpy(m).to(dev), torch.from_numpy(rk).to(dev))
    else: run_epochs(sysm, tr, list(sysm.parameters()), a, fac, attack, dev, a.e4_qat, lambda e: 5e-5, '4-qat', with_loc=True, log=log)
    ex, ps = val_exact(sysm, va, fac_val, dev); log(dict(stage='final-val', exact=ex, psnr=ps, channel_reduction=red))
    torch.save(sysm, os.path.join(a.out, 'final_int8sim.pt'))     # full module (pruned + fake-quant encoder)
    json.dump(dict(cfg=cfg, variant=a.variant, no_bch=a.no_bch, channel_reduction=red), open(os.path.join(a.out, 'meta.json'), 'w'))

if __name__ == '__main__': main()
