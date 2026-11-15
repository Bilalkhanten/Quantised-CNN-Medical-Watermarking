"""Evaluation of one trained run (Tables 10-14 inputs).  Example:
  python evaluate.py --run runs/full_s0 --mias /data/mias [--cross bratsdir lididir ...] --out runs/full_s0/eval.json
Every metric follows Table 9.  Per-image values are stored so that patient-level bootstrap CIs and paired tests can be computed later (stats.py)."""
import argparse, json, os, time, numpy as np, torch, torch.nn.functional as F
import data as D, attacks as A, payload as P
from system import *
from common import *

TABLE10 = [('JPEG', 'Q = 70', lambda x: A.jpeg_real(x, 70)), ('JPEG', 'Q = 50', lambda x: A.jpeg_real(x, 50)),
 ('Gaussian noise', 'σ = 10', lambda x: A.gauss_noise(x, 10)), ('Median filter', '3 × 3', lambda x: A.median(x, 3)), ('Median filter', '5 × 5', lambda x: A.median(x, 5)),
 ('Gaussian blur', 'σ = 2', lambda x: A.gauss_blur(x, 2.0)), ('Rotation', '15°', lambda x: A.affine(x, 15)), ('Scaling', '0.8', lambda x: A.affine(x, 0, 0.8)),
 ('Cropping', '75% kept', lambda x: A.crop_zero(x, .75)), ('Packet loss', '5% of blocks', lambda x: A.packet_loss(x, .05))]
UNSEEN = [('Speckle noise', 'σ² = 0.04', lambda x: A.speckle(x, .04)), ('Sharpening', 'unsharp mask, r = 2', lambda x: A.sharpen(x, 2)), ('Contrast change', 'gain × 1.3', lambda x: A.contrast(x, 1.3)),
 ('JPEG 2000', '0.5 bpp', lambda x: A.jp2k(x, 0.5)), ('Denoising (NLM stand-in)', 'h = 10', lambda x: A.nlm_denoise(x, 10))]
EXTRA = [('Cropping (real crop + resize)', '75% kept', lambda x: A.crop_resize(x, .75)), ('Scaling', '1.2', lambda x: A.affine(x, 0, 1.2)),
 ('Rotation', '5°', lambda x: A.affine(x, 5)), ('JPEG', 'Q = 90', lambda x: A.jpeg_real(x, 90))] + \
 [('JPEG', f'Q = {q}', (lambda q: lambda x: A.jpeg_real(x, q))(q)) for q in (80, 60)] + [('Gaussian noise', f'σ = {s}', (lambda s: lambda x: A.gauss_noise(x, s))(s)) for s in (1, 3, 5, 7)]
SEEN10 = {(n, s) for n, s, _ in TABLE10}
q8c = lambda x: A.q8(x.clamp(0, 1)).detach()

def load_run(run, dev, which='final'):
    """which='final': pruned + fake-quant INT8 model (final_int8sim.pt).  which='fp32': the un-pruned FP32 model after Stage 3 (stage3.pt)."""
    meta = json.load(open(os.path.join(run, 'meta.json')))
    if which == 'final': s = torch.load(os.path.join(run, 'final_int8sim.pt'), weights_only=False, map_location=dev)
    else: s = System(meta['cfg']); s.load_state_dict(torch.load(os.path.join(run, 'stage3.pt'), map_location=dev))
    s.eval(); return s, meta
def conf(logits): return (torch.sigmoid(logits)-.5).abs().mean(1)

@torch.no_grad()
def calibrate_alpha(sysm, loader, fac, dev, target=40.0, iters=12):
    """bisection on the encoder strength alpha so that mean validation PSNR = target (equal-fidelity operating point)"""
    lo, hi = 0.02, 8.0; batches = [(i.to(dev),) + tuple(torch.from_numpy(z).to(dev) for z in fac.batch(i.shape[0])[:2]) for i, _ in loader]
    for _ in range(iters):
        mid = (lo*hi)**.5; sysm.enc.alpha = mid
        ps = torch.cat([psnr(i, sysm.embed(i, m, r)) for i, m, r in batches]).mean().item()
        lo, hi = (mid, hi) if ps > target else (lo, mid)
    sysm.enc.alpha = (lo*hi)**.5; return sysm.enc.alpha

@torch.no_grad()
def eval_payload(sysm, loader, dev, fac, attacks, variant, coded=True, bs=8):
    out = {f'{n}|{s}': dict(ber=[], exact=[], psnr=[], ssim=[]) for n, s, _ in attacks}; fid = dict(psnr=[], ssim=[], psnr_roi=[], patient=[])
    for img, roi in loader:
        img, roi = img.to(dev), roi.to(dev); m, rk, aux = fac.batch(img.shape[0]); mt, rkt = torch.from_numpy(m).to(dev), torch.from_numpy(rk).to(dev)
        iw = sysm.embed(img, mt, rkt); fid['psnr'] += psnr(img, iw).tolist(); fid['ssim'] += ssim(img, iw).tolist()
        fid['psnr_roi'] += [float(v) for v in psnr(img, iw, roi) if roi.sum() > 0]
        for n, s, fn in attacks:
            att = q8c(fn(iw)); r = out[f'{n}|{s}']; r['psnr'] += psnr(iw, att).tolist(); r['ssim'] += ssim(iw, att).tolist()
            hard = (sysm.extract(att) > 0).float(); r['ber'] += ((hard != mt).float().mean(1)*100).tolist()
            hb = hard.cpu().numpy().astype('uint8')
            for b, (key, md, iv) in zip(hb, aux):
                r['exact'].append(float(P.verify(b, key, md, variant) == 'authentic') if coded else float((b[:385] == P.make_payload_bits(key, md, iv, variant)).all()))
    return out, fid

def _loc_metrics(pred, mask, thr=.5):
    p = (F.interpolate(pred, size=mask.shape[-2:], mode='bilinear') > thr).float(); tp = (p*mask).flatten(1).sum(1); fn = ((1-p)*mask).flatten(1).sum(1)
    fp = (p*(1-mask)).flatten(1).sum(1); tn = ((1-p)*(1-mask)).flatten(1).sum(1)
    return dict(dice=dice(p, mask).tolist(), iou=(tp/(tp+fn+fp).clamp_min(1)).tolist(), tdr=(tp/(tp+fn).clamp_min(1)).tolist(), fnr=(fn/(tp+fn).clamp_min(1)).tolist(), fpr=(fp/(fp+tn).clamp_min(1)).tolist())

def matched_filter(img, rk, thr):
    """ablation (x): correlation of the high-passed LL band with R_K; low local correlation = tampered"""
    from models import haar_dwt
    ll = haar_dwt(img)[:, :1]; r = ll-A.gauss_blur(ll, 2.0); c = F.avg_pool2d(r*rk, 9, 1, 4)/(F.avg_pool2d(r*r, 9, 1, 4).sqrt()+1e-6)
    return (c < thr).float()

@torch.no_grad()
def eval_loc(sysm, loader, dev, fac, loc='unet', mf_thr=None):
    kinds = {'copy-move': ('copy',), 'splicing': ('splice',), 'blur/fill': ('blur', 'fill'), 'contrast': ('contrast',)}
    res = {k: {m: [] for m in ('dice', 'iou', 'tdr', 'fnr', 'fpr')} for k in kinds}; scores, labels = [], []
    f = (lambda x, rk: sysm.localise(x, rk)) if loc == 'unet' else (lambda x, rk: matched_filter(x, rk, mf_thr))
    for img, roi in loader:
        img = img.to(dev); m, rk, _ = fac.batch(img.shape[0]); mt, rkt = torch.from_numpy(m).to(dev), torch.from_numpy(rk).to(dev); iw = sysm.embed(img, mt, rkt)
        benign = q8c(A.jpeg_real(iw, 70)); pb = f(benign, rkt); scores += pb.flatten(1).amax(1).tolist(); labels += [0]*img.shape[0]
        for k, kk in kinds.items():
            tmp, mask = A.tamper(iw, img.flip(0), kinds=kk); pr = f(tmp, rkt)
            for mname, v in _loc_metrics(pr, mask).items(): res[k][mname] += v
            scores += pr.flatten(1).amax(1).tolist(); labels += [1]*img.shape[0]
    return res, auc(scores, labels)
def auc(scores, labels):
    s, l = np.asarray(scores), np.asarray(labels); pos, neg = s[l == 1], s[l == 0]
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    return float(((pos[:, None] > neg[None]).sum()+.5*(pos[:, None] == neg[None]).sum())/(len(pos)*len(neg)))

@torch.no_grad()
def eval_auth(sysm, loader, dev, fac, variant):
    """Authentication ROC-AUC: positives = genuine watermarked images after JPEG-70; negatives = unwatermarked images, wrong key, swapped metadata.
    score = digest-match indicator + 0.01 * decoder confidence (the decision itself is the exact digest comparison)."""
    pos, neg = [], []
    for img, roi in loader:
        img = img.to(dev); m, rk, aux = fac.batch(img.shape[0]); mt, rkt = torch.from_numpy(m).to(dev), torch.from_numpy(rk).to(dev)
        iw = sysm.embed(img, mt, rkt); att = q8c(A.jpeg_real(iw, 70))
        for x, kind in ((att, 'pos'), (img, 'unwm'), (att, 'wrongkey'), (att, 'swapmd')):
            lg = sysm.extract(x); h = (lg > 0).cpu().numpy().astype('uint8'); c = conf(lg).cpu().numpy()
            for i, (key, md, iv) in enumerate(aux):
                k2 = os.urandom(16) if kind == 'wrongkey' else key; md2 = aux[(i+1) % len(aux)][1] if kind == 'swapmd' else md
                sc = float(P.verify(h[i], k2, md2, variant) == 'authentic')+0.01*float(c[i]); (pos if kind == 'pos' else neg).append(sc)
    tpr = float(np.mean([s >= 1 for s in pos])); fpr = float(np.mean([s >= 1 for s in neg])); return dict(auc=auc(pos+neg, [1]*len(pos)+[0]*len(neg)), tpr=tpr, fpr=fpr)

@torch.no_grad()
def security_tests(sysm, loader, dev, fac, variant, now_window=300):
    """Threat-model tests of Section 4: wrong key, 1-bit key change, transplant, replay, removal, forgery / re-embedding without K."""
    r = {k: [] for k in ('wrong_key', 'key_1bit', 'transplant_payload_pass', 'transplant_flagged', 'replay_rejected', 'removal_exact', 'forgery_random_bits', 'reembed_no_key')}
    for img, roi in loader:
        img = img.to(dev); n = img.shape[0]
        if n < 2: continue
        m, rk, aux = fac.batch(n); mt, rkt = torch.from_numpy(m).to(dev), torch.from_numpy(rk).to(dev); iw = sysm.embed(img, mt, rkt)
        hb = (sysm.extract(iw) > 0).cpu().numpy().astype('uint8')
        for i, (key, md, iv) in enumerate(aux):
            r['wrong_key'].append(float(P.verify(hb[i], os.urandom(16), md, variant) == 'authentic'))
            k2 = bytearray(key); k2[0] ^= 1; r['key_1bit'].append(float(P.verify(hb[i], bytes(k2), md, variant) == 'authentic'))
            r['forgery_random_bits'].append(float(P.verify(np.random.randint(0, 2, 511).astype('uint8'), key, md, variant) == 'authentic'))
            ts = int(md.split(b'|T')[1]); r['replay_rejected'].append(float(abs((ts+now_window*2)-ts) > now_window))      # freshness policy: replay after 2 windows
        # transplant: add the watermark residual of image i onto image (i+1)
        res = iw-img; tr = q8c(img.roll(1, 0)+res); h2 = (sysm.extract(tr) > 0).cpu().numpy().astype('uint8')
        flag = (sysm.localise(tr, rkt) > .5).flatten(1).float().mean(1) > 0.01
        for i in range(n):
            key, md, iv = aux[i]; r['transplant_payload_pass'].append(float(P.verify(h2[i], key, md, variant) == 'authentic')); r['transplant_flagged'].append(float(flag[i]))
        rm = q8c(A.nlm_denoise(iw)); h3 = (sysm.extract(rm) > 0).cpu().numpy().astype('uint8')
        for i, (key, md, iv) in enumerate(aux): r['removal_exact'].append(float(P.verify(h3[i], key, md, variant) == 'authentic'))
        m2, rk2, aux2 = fac.batch(n); fw = sysm.embed(img, torch.from_numpy(m2).to(dev), torch.from_numpy(rk2).to(dev)); h4 = (sysm.extract(fw) > 0).cpu().numpy().astype('uint8')
        for i, (key, md, iv) in enumerate(aux): r['reembed_no_key'].append(float(P.verify(h4[i], key, md, variant) == 'authentic'))
    return {k: float(np.mean(v)) for k, v in r.items() if v}

def make_loader(recs, bs=8): return torch.utils.data.DataLoader(D.ImageSet(recs), batch_size=bs, shuffle=False)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--mias'); ap.add_argument('--split_seed', type=int, default=0)
    ap.add_argument('--cross', nargs='*', default=[], help='name=path pairs for cross-dataset tests, e.g. BraTS=/data/brats')
    ap.add_argument('--synthetic', type=int, default=0); ap.add_argument('--no_calibrate', action='store_true'); ap.add_argument('--alpha_scale', type=float, default=1.0)
    ap.add_argument('--cross_limit', type=int, default=200); ap.add_argument('--which', default='final', choices=['final', 'fp32']); ap.add_argument('--denoiser', help='path to a torch.save()d / torch.jit denoiser module (1x1xHxW in [0,1] -> same); replaces the NLM stand-in and relabels the row as DnCNN'); ap.add_argument('--out'); ap.add_argument('--seed', type=int, default=123)
    a = ap.parse_args(); dev = get_device(); seed_all(a.seed); sysm, meta = load_run(a.run, dev, a.which); sysm.to(dev); variant, coded = meta['variant'], not meta.get('no_bch')
    if a.synthetic: va, te = (torch.utils.data.DataLoader(D.SyntheticSet(16, seed=s), batch_size=8) for s in (2, 3)); names = None
    else:
        sp = D.patient_split(D.mias_records(a.mias), seed=a.split_seed); va, te = make_loader(sp['val']), make_loader(sp['test']); pat = [r['patient'] for r in sp['test']]
    if a.denoiser:
        net = torch.jit.load(a.denoiser, map_location=dev) if a.denoiser.endswith('.jit') else torch.load(a.denoiser, weights_only=False, map_location=dev); net.eval(); A.DENOISER = net
        UNSEEN[-1] = ('Denoising network', 'DnCNN, σ = 15', lambda x: A.nlm_denoise(x))
    fv = P.PayloadFactory(variant, seed=777, coded=coded); ft = P.PayloadFactory(variant, seed=888, coded=coded)
    if not a.no_calibrate: print('alpha =', calibrate_alpha(sysm, va, fv, dev))
    sysm.enc.alpha *= a.alpha_scale
    res = dict(meta=meta, which=a.which, alpha=float(sysm.enc.alpha), params=dict(enc=count_params(sysm.enc), dec=count_params(sysm.dec), loc=count_params(sysm.loc)))
    pay, fid = eval_payload(sysm, te, dev, ft, TABLE10+UNSEEN+EXTRA, variant, coded); res['attacks'] = pay; res['fidelity'] = fid
    if not a.synthetic: res['fidelity']['patient'] = pat[:len(fid['psnr'])]
    res['loc'], res['loc_auc'] = eval_loc(sysm, te, dev, ft)
    res['auth'] = eval_auth(sysm, te, dev, ft, variant) if coded else None
    res['security'] = security_tests(sysm, te, dev, ft, variant) if coded else None
    # matched-filter localiser ablation (x): threshold picked on the validation split
    best = max(((np.mean(eval_loc(sysm, va, dev, fv, 'mf', t)[0]['copy-move']['dice']), t) for t in (-.2, 0., .1, .2, .3, .4, .5)), key=lambda z: z[0])[1]
    res['loc_matched_filter'], res['loc_matched_filter_auc'] = eval_loc(sysm, te, dev, ft, 'mf', best)
    res['cross'] = {}
    for kv in a.cross:
        name, path = kv.split('='); recs = D.folder_records(path, limit=a.cross_limit); ldr = make_loader(recs); p2, f2 = eval_payload(sysm, ldr, dev, ft, TABLE10, variant, coded)
        l2, la = eval_loc(sysm, ldr, dev, ft); res['cross'][name] = dict(fidelity=f2, attacks=p2, loc=l2, loc_auc=la, auth=eval_auth(sysm, ldr, dev, ft, variant) if coded else None)
    json.dump(res, open(a.out or os.path.join(a.run, 'eval.json'), 'w'))
    mean = lambda v: float(np.mean(v)); print('PSNR %.2f SSIM %.3f' % (mean(fid['psnr']), mean(fid['ssim'])), '| exact (10 seen attacks) %.1f%%' % (100*mean([mean(pay[f'{n}|{s}']['exact']) for n, s, _ in TABLE10])),
          '| Dice(mean of 4 types) %.3f' % mean([mean(v['dice']) for v in res['loc'].values()]))
if __name__ == '__main__': main()
