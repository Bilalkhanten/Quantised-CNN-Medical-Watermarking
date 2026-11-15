"""Statistics of Table 9: patient-level bootstrap CIs, paired Wilcoxon with Holm correction, mean +- std over seeds.
  python stats.py summary runs/full_s*/eval.json                       # mean +- std over seeds + patient-bootstrap 95% CI per attack
  python stats.py compare runs/full_s0/eval.json runs/base_dwtsvd.json # paired Wilcoxon (per-image BER) per attack, Holm-corrected"""
import sys, json, numpy as np
from scipy import stats as ss

def patient_bootstrap(vals, patients, n=2000, seed=0, q=(2.5, 97.5)):
    """resample *patients* (not images) with replacement -> CI of the mean"""
    v, p = np.asarray(vals, float), np.asarray(patients); ids = np.unique(p); rng = np.random.default_rng(seed)
    per = {i: v[p == i] for i in ids}; means = []
    for _ in range(n):
        s = rng.choice(ids, len(ids)); means.append(np.concatenate([per[i] for i in s]).mean())
    return tuple(np.percentile(means, q))
def holm(pvals):
    p = np.asarray(pvals, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m-r)*p[i]); adj[i] = min(1.0, run)
    return adj
def paired_wilcoxon(a, b):
    d = np.asarray(a)-np.asarray(b)
    return 1.0 if np.all(d == 0) else float(ss.wilcoxon(a, b, zero_method='zsplit').pvalue)

def summary(paths):
    runs = [json.load(open(p)) for p in paths]; keys = list(runs[0]['attacks'])
    print('%-34s %-18s %-22s' % ('attack', 'BER % (mean±sd seeds)', 'exact % (mean±sd seeds)  [patient-bootstrap CI, seed 0]'))
    pat = runs[0]['fidelity'].get('patient')
    for k in keys:
        ber = [np.mean(r['attacks'][k]['ber']) for r in runs]; ex = [100*np.mean(r['attacks'][k]['exact']) for r in runs]; ci = ''
        if pat is not None and len(pat) == len(runs[0]['attacks'][k]['exact']):
            lo, hi = patient_bootstrap(np.array(runs[0]['attacks'][k]['exact'])*100, pat); ci = '[%.1f, %.1f]' % (lo, hi)
        print('%-34s %6.2f ± %-9.2f %6.1f ± %-8.1f %s' % (k, np.mean(ber), np.std(ber), np.mean(ex), np.std(ex), ci))
def compare(a_path, b_path, metric='ber'):
    A, B = json.load(open(a_path)), json.load(open(b_path)); keys = [k for k in A['attacks'] if k in B['attacks']]
    p = [paired_wilcoxon(A['attacks'][k][metric], B['attacks'][k][metric]) for k in keys]; adj = holm(p)
    print('%-34s %9s %9s %10s %10s' % ('attack', 'A '+metric, 'B '+metric, 'p', 'p (Holm)'))
    for k, pi, ai in zip(keys, p, adj): print('%-34s %9.3f %9.3f %10.2e %10.2e %s' % (k, np.mean(A['attacks'][k][metric]), np.mean(B['attacks'][k][metric]), pi, ai, '*' if ai < .05 else ''))
if __name__ == '__main__':
    if sys.argv[1] == 'summary': summary(sys.argv[2:])
    elif sys.argv[1] == 'compare': compare(*sys.argv[2:4], *(sys.argv[4:5] or []))
