"""Diagnostic-impact test (Table 11 / Reviewer comment on clinical utility): does watermarking change a downstream classifier's ROC-AUC?
A small CNN is trained on ORIGINAL training images to separate images with a marked abnormality from normal ones (MIAS Info.txt), then scored on the
ORIGINAL and on the WATERMARKED test images; paired DeLong test on the two AUCs.   python diag_classifier.py --run runs/full_s0 --mias /data/mias"""
import argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy import stats as ss
import data as D, evaluate as E, payload as P
from common import *

def _midrank(x):
    o = np.argsort(x); z = x[o]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]: j += 1
        t[i:j] = 0.5*(i+j-1)+1; i = j
    out = np.empty(n); out[o] = t; return out
def delong_paired(y, s1, s2):
    """paired DeLong test for two correlated ROC-AUCs (Sun & Xu 2014). y in {0,1}; returns auc1, auc2, p"""
    y = np.asarray(y); pos, neg = y == 1, y == 0; m, n = pos.sum(), neg.sum(); V10, V01, aucs = [], [], []
    for s in (np.asarray(s1, float), np.asarray(s2, float)):
        tx, ty, tz = _midrank(s[pos]), _midrank(s[neg]), _midrank(s)
        aucs.append((tz[pos].sum()-m*(m+1)/2)/(m*n)); V10.append((tz[pos]-tx)/n); V01.append(1-(tz[neg]-ty)/m)
    S10, S01 = np.cov(np.array(V10)), np.cov(np.array(V01)); S = S10/m+S01/n; L = np.array([1, -1])
    var = float(L@S@L); z = (aucs[0]-aucs[1])/np.sqrt(var) if var > 0 else 0.0
    return aucs[0], aucs[1], float(2*ss.norm.sf(abs(z)))

class SmallCNN(nn.Module):
    def __init__(s, C=16):
        super().__init__(); L = []; c = 1
        for o in (C, 2*C, 4*C, 4*C, 8*C): L += [nn.Conv2d(c, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]; c = o
        s.net = nn.Sequential(*L); s.fc = nn.Linear(c, 1)
    def forward(s, x): return s.fc(F.adaptive_avg_pool2d(s.net(x), 1).flatten(1)).squeeze(1)

@torch.no_grad()
def scores(clf, xs, dev, bs=16):
    clf.eval(); return torch.cat([clf(x.to(dev)).cpu() for x in xs.split(bs)]).numpy()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--mias'); ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--synthetic', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out'); a = ap.parse_args(); dev = get_device(); seed_all(a.seed); sysm, meta = E.load_run(a.run, dev); sysm.to(dev)
    if a.synthetic: tr, va, te = (D.SyntheticSet(a.synthetic, seed=s) for s in (1, 2, 3))
    else: sp = D.patient_split(D.mias_records(a.mias), seed=0); tr, va, te = (D.ImageSet(sp[k], train=(k == 'train')) for k in ('train', 'val', 'test'))
    def XY(ds):
        x = torch.stack([ds[i][0] for i in range(len(ds))]); r = torch.stack([ds[i][1] for i in range(len(ds))])
        y = (r.flatten(1).sum(1) > 0).float() if not a.synthetic else (x.flatten(1).mean(1) > x.flatten(1).mean(1).median()).float(); return x, y
    (xt, yt), (xv, yv), (xe, ye) = XY(tr), XY(va), XY(te); clf = SmallCNN().to(dev); opt = torch.optim.AdamW(clf.parameters(), 1e-3, weight_decay=1e-4)
    pw = ((1-yt).sum()/yt.sum().clamp_min(1)).to(dev)
    for ep in range(a.epochs):
        clf.train(); perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 16):
            idx = perm[i:i+16]; x = xt[idx].to(dev); y = yt[idx].to(dev); x = torch.where(torch.rand(len(idx), 1, 1, 1, device=dev) < .5, x.flip(-1), x)
            L = F.binary_cross_entropy_with_logits(clf(x), y, pos_weight=pw); opt.zero_grad(); L.backward(); opt.step()
    fac = P.PayloadFactory(meta['variant'], seed=999); fv = P.PayloadFactory(meta['variant'], seed=777)
    E.calibrate_alpha(sysm, torch.utils.data.DataLoader(va, batch_size=8), fv, dev)
    with torch.no_grad():
        xw = []
        for x in xe.split(8):
            m, rk, _ = fac.batch(x.shape[0]); xw.append(sysm.embed(x.to(dev), torch.from_numpy(m).to(dev), torch.from_numpy(rk).to(dev)).cpu())
        xw = torch.cat(xw)
    y = ye.numpy(); s0, s1 = scores(clf, xe, dev), scores(clf, xw, dev); auc0, auc1, p = delong_paired(y, s0, s1)
    r = dict(auc_original=auc0, auc_watermarked=auc1, delta=auc1-auc0, p_delong=p, n_test=len(y), n_abnormal=int(y.sum()), mean_psnr=float(psnr(xe, xw).mean()))
    print(r); json.dump(r, open(a.out or a.run+'/diag.json', 'w'))
if __name__ == '__main__': main()
