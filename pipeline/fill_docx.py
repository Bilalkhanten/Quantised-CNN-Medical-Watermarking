"""Write MEASURED results into Tables 10-14 of the manuscript (docx tables 9-13).  Cells without a supplied result are set to 'n/m' (not measured) -- an old illustrative
number is never left in place next to measured ones.  Updated cells are highlighted yellow (--no_highlight to disable).  'Illustrative' captions are stripped per table
ONLY when every cell of that table was filled from a measurement (--strip_labels); body-text paragraphs that still mention illustrative values are listed for manual editing.

  python fill_docx.py --docx Revised_Manuscript.docx --out Revised_Manuscript_measured.docx \\
     --full runs/full_s0/eval.json runs/full_s1/eval.json ...          # pruned INT8 model, one eval.json per seed
     --fp32 runs/fp32_s0/eval.json ...                                  # evaluate.py --which fp32 (un-pruned FP32 model after Stage 3)
     --baseline dwtsvd=runs/base_dwtsvd.json hidden=runs/base_hidden.json redmark=... mbrs=...
     --abl noattack=runs/abl_noattack/eval.json nobch=... nostn=... noroi=... spatial=... globalpool=... nork=... nod=... ptq=... nofinetune=... alpha07=... alpha13=...
     --cost runs/full_s0/cost.json                                      # measure_cost.py on the target device
Cross-dataset rows come from the `cross` entries of the eval.json files (evaluate.py --cross CBIS=... CXR14=... BraTS=... LIDC=...)."""
import argparse, json, re, numpy as np, docx
from docx.enum.text import WD_COLOR_INDEX
import evaluate as E

GRID = [f'{n}|{s}' for n, s, _ in E.TABLE10]
ATT_ROWS = {  # docx row label -> eval.json key(s) (first present is used)
 ('JPEG', 'Q = 70'): ['JPEG|Q = 70'], ('JPEG', 'Q = 50'): ['JPEG|Q = 50'], ('Gaussian noise', 'σ = 10'): ['Gaussian noise|σ = 10'], ('Median filter', '3 × 3'): ['Median filter|3 × 3'],
 ('Median filter', '5 × 5'): ['Median filter|5 × 5'], ('Gaussian blur', 'σ = 2'): ['Gaussian blur|σ = 2'], ('Rotation', '15°'): ['Rotation|15°'], ('Scaling', '0.8'): ['Scaling|0.8'],
 ('Cropping', '75% kept'): ['Cropping|75% kept'], ('Packet loss', '5% of blocks'): ['Packet loss|5% of blocks'], ('Speckle noise', 'σ² = 0.04'): ['Speckle noise|σ² = 0.04'],
 ('Sharpening', 'unsharp mask, r = 2'): ['Sharpening|unsharp mask, r = 2'], ('Contrast change', 'gain × 1.3'): ['Contrast change|gain × 1.3'], ('JPEG 2000', '0.5 bpp'): ['JPEG 2000|0.5 bpp'],
 ('Denoising network', 'DnCNN, σ = 15'): ['Denoising network|DnCNN, σ = 15', 'Denoising (NLM stand-in)|h = 10']}
ABL_ORDER = ['fp32', 'noattack', 'nobch', 'nostn', 'noroi', 'spatial', 'full', 'alpha', 'globalpool', 'nork', 'mf', 'ptq', 'nofinetune', 'nod']   # docx rows 1..14
CROSS = [('MIAS', None), ('CBIS', 'CBIS'), ('CXR14', 'CXR14'), ('BraTS', 'BraTS'), ('LIDC', 'LIDC')]

ld = lambda ps: [json.load(open(p)) for p in ps]
def grid_exact(r, cross=None):
    a = (r['cross'][cross]['attacks'] if cross else r['attacks']); return 100*np.mean([np.mean(a[k]['exact']) for k in GRID])
def dice_mean(r, cross=None, key='loc'):
    L = r['cross'][cross]['loc'] if cross else r[key]; return float(np.mean([np.mean(v['dice']) for v in L.values()]))
def fid(r, cross=None): return r['cross'][cross]['fidelity'] if cross else r['fidelity']
def auth_auc(r, cross=None): a = r['cross'][cross]['auth'] if cross else r['auth']; return a['auc'] if a else None
def grid_ber(r, cross=None):
    a = r['cross'][cross]['attacks'] if cross else r['attacks']; v = [np.mean(a[k]['ber']) for k in GRID if k in a and a[k].get('ber')]
    return float(np.mean(v)) if v else None
def ms(vals, f, pm=True):
    v = np.asarray(vals, float); s = f % v.mean(); return s + (' ± ' + f % v.std(ddof=1) if pm and len(v) > 1 else '')

class Filler:
    def __init__(s, doc, hl):
        s.doc, s.hl, s.filled, s.missing = doc, hl, {}, {}
        # locate Table 10 by its header so that tables added elsewhere in the manuscript do not shift the mapping
        idx = [i for i, t in enumerate(doc.tables) if t.rows[0].cells[0].text.strip() == 'Attack' and 'Exact' in ' '.join(c.text for c in t.rows[0].cells)]
        s.off = (idx[0] - 9) if idx else 0
    def set(s, t, r, c, text, tid):
        cell = s.doc.tables[t+s.off].rows[r].cells[c]; p = cell.paragraphs[0]
        for extra in cell.paragraphs[1:]: extra._p.getparent().remove(extra._p)
        if p.runs:
            p.runs[0].text = text
            for x in p.runs[1:]: x._r.getparent().remove(x._r)
        else: p.add_run(text)
        if s.hl: p.runs[0].font.highlight_color = WD_COLOR_INDEX.YELLOW
        (s.missing if text == 'n/m' else s.filled).setdefault(tid, []).append((r, c))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--docx', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--full', nargs='*', default=[]); ap.add_argument('--fp32', nargs='*', default=[]); ap.add_argument('--baseline', nargs='*', default=[]); ap.add_argument('--abl', nargs='*', default=[])
    ap.add_argument('--cost'); ap.add_argument('--no_highlight', action='store_true'); ap.add_argument('--strip_labels', action='store_true'); a = ap.parse_args()
    doc = docx.Document(a.docx); F = Filler(doc, not a.no_highlight); full, fp32 = ld(a.full), ld(a.fp32)
    base = {kv.split('=')[0]: json.load(open(kv.split('=')[1])) for kv in a.baseline}; abl = {kv.split('=')[0]: ld([kv.split('=')[1]]) for kv in a.abl}
    NM = 'n/m'
    # ---------------- Table 10 (docx 9): attack severity + recovery, pooled over seeds
    if full:
        for r in range(1, len(doc.tables[9+F.off].rows)):
            key = (doc.tables[9+F.off].rows[r].cells[0].text.strip(), doc.tables[9+F.off].rows[r].cells[1].text.strip())
            k = next((kk for kk in ATT_ROWS.get(key, []) if kk in full[0]['attacks']), None)
            if k is None:
                [F.set(9, r, c, NM, 'T10') for c in (2, 3, 4, 5)]; continue
            pool = lambda f: np.concatenate([np.asarray(x['attacks'][k][f]) for x in full])
            F.set(9, r, 2, ms(pool('psnr'), '%.1f'), 'T10'); F.set(9, r, 3, ms(pool('ssim'), '%.3f'), 'T10'); F.set(9, r, 4, ms(pool('ber'), '%.1f'), 'T10'); F.set(9, r, 5, '%.1f' % (100*pool('exact').mean()), 'T10')
            if k.startswith('Denoising (NLM'): F.set(9, r, 0, 'Denoising (non-local means)', 'T10'); F.set(9, r, 1, 'h = 10 (DnCNN not run)', 'T10')
    # ---------------- Table 11 (docx 10): method comparison
    cost = json.load(open(a.cost)) if a.cost else None
    def node_energy(): 
        if cost and cost.get('energy_mJ_per_exec'): e = cost['energy_mJ_per_exec']; return '%.1f' % (e.get('payload_encode', 0)+e.get('onnx_enc_int8', 0))
        return NM
    def node_latency():
        if cost and all(k in cost for k in ('payload_encode', 'onnx_enc_int8')): return '%.1f' % (cost['payload_encode']['median_ms']+cost['onnx_enc_int8']['median_ms'])
        return NM
    rows = {1: ('dwtsvd', None), 2: ('hidden', None), 3: ('redmark', None), 4: ('mbrs', None), 5: ('prop_fp32', fp32), 6: ('prop_int8', full)}
    for r, (name, runs) in rows.items():
        if runs is None:
            b = base.get(name)
            if not b: [F.set(10, r, c, NM, 'T11') for c in range(1, 10)]; continue
            f = b['fidelity']; F.set(10, r, 1, '%.1f' % np.mean(f['psnr']), 'T11'); F.set(10, r, 2, '%.3f' % np.mean(f['ssim']), 'T11'); F.set(10, r, 3, '%.1f' % (100*np.mean([np.mean(b['attacks'][k]['exact']) for k in GRID])), 'T11')
            F.set(10, r, 4, '—', 'T11'); F.set(10, r, 5, NM, 'T11'); F.set(10, r, 6, '%.3f' % (b.get('params', {}).get('enc', 0)/1e6) if b.get('params', {}).get('enc') else '—', 'T11'); F.set(10, r, 7, NM, 'T11')
            bb = [np.mean(b['attacks'][k]['ber']) for k in GRID if k in b['attacks'] and b['attacks'][k].get('ber')]; F.set(10, r, 8, '%.2f' % np.mean(bb) if bb else NM, 'T11'); F.set(10, r, 9, NM, 'T11')
        else:
            if not runs: [F.set(10, r, c, NM, 'T11') for c in range(1, 10)]; continue
            F.set(10, r, 1, ms([np.mean(fid(x)['psnr']) for x in runs], '%.1f', pm=False), 'T11'); F.set(10, r, 2, ms([np.mean(fid(x)['ssim']) for x in runs], '%.3f', pm=False), 'T11')
            F.set(10, r, 3, ms([grid_exact(x) for x in runs], '%.1f'), 'T11'); F.set(10, r, 4, ms([dice_mean(x) for x in runs], '%.2f', pm=False), 'T11')
            F.set(10, r, 5, ms([auth_auc(x) for x in runs if auth_auc(x) is not None], '%.3f', pm=False), 'T11'); F.set(10, r, 6, '%.3f' % (runs[0]['params']['enc']/1e6), 'T11')
            F.set(10, r, 7, node_energy() if name == 'prop_int8' else NM, 'T11')
            bv = [grid_ber(x) for x in runs if grid_ber(x) is not None]; F.set(10, r, 8, ms(bv, '%.2f') if bv else NM, 'T11')
            F.set(10, r, 9, node_latency() if name == 'prop_int8' else NM, 'T11')
    # ---------------- Table 12 (docx 11): ablations
    def abl_row(r, runs, rng=None):
        if not runs: [F.set(11, r, c, NM, 'T12') for c in range(1, 5)]; return
        if rng:
            P_ = [np.mean(fid(x)['psnr']) for x in runs]; X = [grid_exact(x) for x in runs]; D_ = [dice_mean(x) for x in runs]; A_ = [auth_auc(x) for x in runs if auth_auc(x) is not None]
            F.set(11, r, 1, '%.1f–%.1f' % (max(P_), min(P_))); F.set(11, r, 2, '%.1f–%.1f' % (min(X), max(X))); F.set(11, r, 3, '%.2f–%.2f' % (min(D_), max(D_))); F.set(11, r, 4, '%.3f–%.3f' % (min(A_), max(A_))); return
        F.set(11, r, 1, ms([np.mean(fid(x)['psnr']) for x in runs], '%.1f', pm=False), 'T12'); F.set(11, r, 2, ms([grid_exact(x) for x in runs], '%.1f', pm=False), 'T12')
        F.set(11, r, 3, ms([dice_mean(x) for x in runs], '%.2f', pm=False), 'T12'); au = [auth_auc(x) for x in runs if auth_auc(x) is not None]; F.set(11, r, 4, ms(au, '%.3f', pm=False) if au else NM, 'T12')
    def abl_range(r):
        runs = abl.get('alpha07', [])+abl.get('alpha13', [])
        if not runs: [F.set(11, r, c, NM, 'T12') for c in range(1, 5)]; return
        P_ = [np.mean(fid(x)['psnr']) for x in runs]; X = [grid_exact(x) for x in runs]; D_ = [dice_mean(x) for x in runs]; A_ = [auth_auc(x) for x in runs if auth_auc(x) is not None]
        F.set(11, r, 1, '%.1f–%.1f' % (max(P_), min(P_)), 'T12'); F.set(11, r, 2, '%.1f–%.1f' % (min(X), max(X)), 'T12'); F.set(11, r, 3, '%.2f–%.2f' % (min(D_), max(D_)), 'T12'); F.set(11, r, 4, '%.3f–%.3f' % (min(A_), max(A_)) if A_ else NM, 'T12')
    for r, key in enumerate(ABL_ORDER, start=1):
        if key == 'fp32': abl_row(r, fp32)
        elif key == 'full': abl_row(r, full)
        elif key == 'alpha': abl_range(r)
        elif key == 'mf':
            if not fp32: [F.set(11, r, c, NM, 'T12') for c in range(1, 5)]; continue
            F.set(11, r, 1, ms([np.mean(fid(x)['psnr']) for x in fp32], '%.1f', pm=False), 'T12'); F.set(11, r, 2, ms([grid_exact(x) for x in fp32], '%.1f', pm=False), 'T12')
            F.set(11, r, 3, ms([dice_mean(x, key='loc_matched_filter') for x in fp32], '%.2f', pm=False), 'T12'); F.set(11, r, 4, ms([auth_auc(x) for x in fp32 if auth_auc(x) is not None], '%.3f', pm=False), 'T12')
        else: abl_row(r, abl.get(key, []))
    # ---------------- Table 13 (docx 12): cost on the target device
    stages = {1: ('payload_encode', None), 2: ('dwt_idwt', None), 3: ('onnx_enc_fp32', 'dwt_idwt'), 4: ('onnx_enc_int8', 'dwt_idwt')}
    if cost:
        en = cost.get('energy_mJ_per_exec', {})
        for r, (k, sub) in stages.items():
            if k not in cost: [F.set(12, r, c, NM, 'T13') for c in (1, 2, 3)]; continue
            med, p95 = cost[k]['median_ms'], cost[k]['p95_ms']
            if sub: med, p95 = max(med-cost[sub]['median_ms'], 0), max(p95-cost[sub]['p95_ms'], 0)         # encoder graph includes Haar DWT/IDWT -> report the encoder body only
            F.set(12, r, 1, '%.1f / %.1f' % (med, p95), 'T13')
            if k in en: F.set(12, r, 2, '%.1f' % en[k], 'T13'); F.set(12, r, 3, '%.0f' % (1e3*en[k]/(cost[k]['median_ms']) if cost[k]['median_ms'] else 0), 'T13')
            else: F.set(12, r, 2, NM, 'T13'); F.set(12, r, 3, NM, 'T13')
        if all(k in cost for k in ('payload_encode', 'onnx_enc_int8')):
            F.set(12, 6, 1, '%.1f / %.1f' % (cost['payload_encode']['median_ms']+cost['onnx_enc_int8']['median_ms'], cost['payload_encode']['p95_ms']+cost['onnx_enc_int8']['p95_ms']), 'T13')
            F.set(12, 6, 2, '%.1f' % (en['payload_encode']+en['onnx_enc_int8']) if 'payload_encode' in en and 'onnx_enc_int8' in en else NM, 'T13'); F.set(12, 6, 3, NM, 'T13')      # peak-memory column stays analytic (activation + weight sizes); process RSS is not comparable
    else: [F.set(12, r, c, NM, 'T13') for r in (1, 2, 3, 4, 6) for c in (1, 2, 3)]
    # ---------------- Table 14 (docx 13): cross-dataset
    for r, (nm, ck) in enumerate(CROSS, start=1):
        if not full or (ck and ck not in full[0].get('cross', {})): [F.set(13, r, c, NM, 'T14') for c in range(1, 5)]; continue
        F.set(13, r, 1, ms([np.mean(fid(x, ck)['psnr']) for x in full], '%.1f'), 'T14'); F.set(13, r, 2, ms([grid_exact(x, ck) for x in full], '%.1f'), 'T14'); F.set(13, r, 3, ms([dice_mean(x, ck) for x in full], '%.2f'), 'T14')
        au = [auth_auc(x, ck) for x in full if auth_auc(x, ck) is not None]; F.set(13, r, 4, ms(au, '%.3f', pm=False) if au else NM, 'T14')
    # ---------------- captions
    complete = {t: t in F.filled and t not in F.missing for t in ('T10', 'T11', 'T12', 'T13', 'T14')}
    pats = [r'\s*\(?(?:illustrative demonstration values,? not measurements)\)?[.;]?', r'; payload-recovery columns are illustrative demonstration values, not measurements', r'\bAll values are illustrative demonstration values, not measurements\.\s*']
    leftover = []
    for p in doc.paragraphs:
        m = re.match(r'\s*Table (\d+)\.', p.text); t = f'T{m.group(1)}' if m else None
        if a.strip_labels and t and complete.get(t):
            for run in p.runs:
                for pt in pats: run.text = re.sub(pt, '', run.text)
        if 'illustrative' in p.text.lower(): leftover.append(p.text[:110])
    doc.save(a.out)
    print('filled cells per table:', {k: len(v) for k, v in F.filled.items()}); print('NOT measured (n/m) per table:', {k: len(v) for k, v in F.missing.items()})
    print('tables fully measured:', [k for k, v in complete.items() if v]); print('paragraphs that still say "illustrative" (edit by hand):'); [print('  -', s) for s in leftover]
if __name__ == '__main__': main()
