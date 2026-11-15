"""Figures that depend on the payload code or on image data (Comment 8: every figure must say what it is).
  python make_figures.py conceptual OUTDIR           # Fig 3 (computed from payload.py) and Fig 7 (flow chart), AES-CBC and HMAC variants
  python make_figures.py fig11_standin OUTDIR        # Fig 11 on bundled pydicom samples + phantom, NON-trained stand-in embedding (documented sources)
  python make_figures.py fig11 OUTDIR --run runs/full_s0 --mias DIR [--n 6]   # final Fig 11: MIAS test slices + the TRAINED encoder (calibrated to 40 dB)"""
import sys, os, argparse, numpy as np, torch, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mp
import payload as P
from models import haar_dwt, haar_idwt, detail_energy
BOX = dict(boxstyle='round,pad=0.02,rounding_size=0.01', lw=1.8, ec='black'); CR, GR, BL, RD, YL, GN = '#FDEBD0', '#F2F2F2', '#CFE2F3', '#F8D7D7', '#FFF6C9', '#D8F2CA'

def box(ax, x, y, w, h, text, fc, fs=15, bold=False):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, fc=fc, **BOX)); ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', linespacing=1.35)
def arrow(ax, x0, y0, x1, y1): ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', lw=1.8, color='black'))

def fig3(out):
    key = bytes(range(16)); md = b'PID00012345|DEV0042|T1700000000'; iv = bytes(range(100, 116)); fig = plt.figure(figsize=(19.8, 10.5), dpi=100)
    ax = fig.add_axes([0, 0.52, 1, 0.48]); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis('off')
    ax.text(50, 47.5, 'Payload generation (acquisition node), Eqs. (7)–(9); two variants of the authentication step', ha='center', fontsize=21, fontweight='bold')
    box(ax, 1, 20, 12, 15, 'Metadata M\n(UTF-8,\nfixed length)', GR, 15)
    box(ax, 17, 26, 24, 12, 'Variant A (baseline)\nSHA-256 → AES-128-CBC, random IV\nd: 256 bits → c: 256 bits', CR, 13); box(ax, 17, 9, 24, 12, 'Variant B (HMAC)\nτ = HMAC-SHA-256(K_mac, IV ‖ M)\nτ: 256 bits', CR, 13)
    box(ax, 45, 17, 14, 15, 'IV ‖ c   (A)\nIV ‖ τ   (B)\n\n384 bits', CR, 14); box(ax, 63, 17, 16, 15, 'BCH(511, 385)\nt = 14\n\n511 bits', CR, 14); box(ax, 83, 17, 16, 15, 'Keyed\npermutation π_K\n\nm: 511 bits', CR, 14)
    arrow(ax, 13, 27.5, 17, 32); arrow(ax, 13, 27.5, 17, 15); arrow(ax, 41, 32, 45, 27); arrow(ax, 41, 15, 45, 22); arrow(ax, 59, 24.5, 63, 24.5); arrow(ax, 79, 24.5, 83, 24.5)
    ax.text(1, 5.5, 'Keys K_enc (AES), K_mac (HMAC) and K (permutation, R_K) are derived from a provisioned master key. The 385th data bit is zero padding. Variant B is the authenticated-integrity variant of Section 4.', fontsize=13.5)
    ax.text(1, 1.5, 'Bit budget (both variants): 128 (IV) + 256 (ciphertext or tag) = 384 data bits, + 1 pad = 385, + 126 BCH parity = 511 bits = 0.0078 bpp for 256 × 256.', fontsize=13.5)
    def codeword(v): msg = P.make_payload_bits(key, md, iv, v); return np.asarray(P._bch.encode(P.galois.GF2(msg)), np.uint8)
    for k, (v, lab) in enumerate((('cbc', 'A: AES-128-CBC'), ('hmac', 'B: HMAC-SHA-256'))):
        a = fig.add_axes([0.07, 0.355-0.19*k, 0.40, 0.11]); a.imshow(codeword(v)[None], cmap='gray_r', aspect='auto', interpolation='nearest'); a.set_yticks([]); a.set_xticks([0, 128, 384, 510]); a.set_xticklabels(['0', '128', '384', '511'], fontsize=13)
        for xx in (128, 384): a.axvline(xx-.5, color='#B00000', lw=1.5)
        for xx, t in ((64, 'IV'), (256, 'ciphertext' if v == 'cbc' else 'tag'), (448, 'pad + parity')): a.text(xx, 0, t, color='#B00000', ha='center', va='center', fontsize=13, bbox=dict(fc='white', ec='none', pad=1))
        a.set_title(f'(a{k+1}) Real coded bits before permutation, variant {lab} (example key, IV)', fontsize=14, loc='left')
    for k, (ivv, t) in enumerate(((iv, '(b) R_K, 128 × 128 (AES-CTR keystream)'), (bytes([iv[0] ^ 1])+iv[1:], '(c) R_K after a 1-bit IV change'))):
        a = fig.add_axes([0.58+0.215*k, 0.03, 0.19, 0.34]); a.imshow(P.reference_pattern(P.derive(key)['rk'], ivv), cmap='gray', interpolation='nearest'); a.axis('off'); a.set_title(t, fontsize=14)
    fig.savefig(out, dpi=100); plt.close(fig)

def poly(ax, pts):
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]): ax.plot([x0, x1], [y0, y1], color='black', lw=1.8, solid_capstyle='butt')
    arrow(ax, *pts[-2], *pts[-1])

def fig7(out):
    fig = plt.figure(figsize=(19.8, 11.7), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis('off')
    ax.text(1, 55.5, 'Verification and decision protocol (receiver)', fontsize=22, fontweight='bold')
    box(ax, 1.5, 40, 16, 10, 'Received image,\nmetadata M′,\nkey', GR, 15); box(ax, 23, 40.5, 15, 9, 'Decoder D_φ\n511 soft bits', BL, 15, True); box(ax, 42, 40.5, 15, 9, 'Un-permute,\nBCH decode', CR, 15)
    d = mp.Polygon([[61, 45], [70, 51], [79, 45], [70, 39]], fc=YL, **{k: v for k, v in BOX.items() if k != 'boxstyle'}); ax.add_patch(d); ax.text(70, 45, '> 14 errors?', ha='center', va='center', fontsize=14)
    box(ax, 84, 40.5, 14, 9, 'Unreadable', RD, 17); arrow(ax, 17.5, 45, 23, 45); arrow(ax, 38, 45, 42, 45); arrow(ax, 57, 45, 61, 45); arrow(ax, 79, 45, 84, 45); ax.text(80, 46.3, 'yes', fontsize=14); ax.text(71, 36.5, 'no', fontsize=14); arrow(ax, 70, 39, 70, 33)
    box(ax, 56, 24, 40, 9, 'Variant A: AES decrypt → d̂;  d′ = SHA-256(M′)\nVariant B: recovered tag;  τ′ = HMAC-SHA-256(K_mac, IV ‖ M′)', CR, 14)
    d2 = mp.Polygon([[24, 28.5], [34, 35], [44, 28.5], [34, 22]], fc=YL, **{k: v for k, v in BOX.items() if k != 'boxstyle'}); ax.add_patch(d2); ax.text(34, 28.5, 'd̂ = d′ ?\n(B: tag = τ′ ?)', ha='center', va='center', fontsize=13.5)
    arrow(ax, 56, 28.5, 44, 28.5); box(ax, 1.5, 24, 15, 9, 'Metadata mismatch,\nwrong key or forgery', RD, 14); arrow(ax, 24, 28.5, 16.5, 28.5); ax.text(19, 30, 'no', fontsize=14); arrow(ax, 34, 22, 34, 19.2); ax.plot([34, 10], [19.2, 19.2], color='black', lw=1.8); ax.text(35, 20.3, 'yes', fontsize=14)
    box(ax, 1.5, 6, 18, 11, 'Localiser L_ψ with\nR_K regenerated', BL, 15, True); box(ax, 24, 7, 17, 9, 'Threshold τ, area\n≥ a_min → regions', GR, 14); arrow(ax, 19.5, 11.5, 24, 11.5); arrow(ax, 10, 19.2, 10, 17); arrow(ax, 41, 11.5, 47, 11.5)
    d3 = mp.Polygon([[47, 11.5], [54, 17], [61, 11.5], [54, 6]], fc=YL, **{k: v for k, v in BOX.items() if k != 'boxstyle'}); ax.add_patch(d3); ax.text(54, 11.5, 'regions\nflagged?', ha='center', va='center', fontsize=13.5)
    box(ax, 72, 12.5, 26, 7.5, 'Authentic', GN, 18, True); box(ax, 72, 2.5, 26, 8, 'Genuine source, locally\nmodified (regions shown)', '#FFE9A8', 14); poly(ax, [(61, 11.5), (66, 11.5), (66, 16.25), (72, 16.25)]); ax.text(62.2, 12.4, 'no', fontsize=14); poly(ax, [(54, 6), (54, 4.5), (72, 4.5)]); ax.text(56.5, 3.2, 'yes', fontsize=14)
    ax.text(1, 1, 'Decisions are exact (digest or tag match), not similarity scores. Outcomes correspond to Table 8.', fontsize=14, style='italic'); fig.savefig(out, dpi=100); plt.close(fig)

def patch_fig1(src, out):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(src).convert('RGB'); dr = ImageDraw.Draw(im); F = lambda s: ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', s)
    def repaint(x0, y0, x1, y1, lines, fs):
        fill = im.getpixel((x0+3, y0+3)); dr.rectangle([x0, y0, x1, y1], fill=fill); n = len(lines); h = (y1-y0)/n
        for i, t in enumerate(lines): dr.text(((x0+x1)/2, y0+h*(i+.5)), t, font=F(fs), fill='black', anchor='mm')
    repaint(51, 262, 548, 338, ['M → SHA-256 → AES-CBC  /  HMAC', '→ BCH → π_K        m: 511 bits'], 21)
    repaint(1580, 1136, 1932, 1262, ['Un-permute, BCH,', 'AES or HMAC check,', 'compare digest / tag'], 27); im.save(out)

def _standin(img01, target=40.0):
    x = torch.from_numpy(img01).float()[None, None]; y = haar_dwt(x); D = detail_energy(y)[0, 0]; rk = torch.from_numpy(P.reference_pattern(P.derive(bytes(range(16)))['rk'], bytes(range(16)))); lo, hi = 1e-4, 0.5
    def emb(c): y2 = torch.cat([y[:, :1]+c*(rk*(0.3+D))[None, None], y[:, 1:]], 1); return ((haar_idwt(y2).clamp(0, 1)*255).round()/255)
    for _ in range(30):
        mid = (lo*hi)**.5; w = emb(mid); ps = 10*torch.log10(1/((w-x)**2).mean().clamp_min(1e-12)).item(); lo, hi = (mid, hi) if ps > target else (lo, mid)
    w = emb((lo*hi)**.5); return x[0, 0].numpy(), w[0, 0].numpy()
def _grid(items, labels, out, row_labels, amp=50):
    fig, axs = plt.subplots(3, len(items), figsize=(19.8, 11.25), dpi=100, squeeze=False); fig.subplots_adjust(left=.07, right=.995, top=.93, bottom=.02, wspace=.03, hspace=.06)
    for j, (o, w) in enumerate(items):
        for i, (a, cm) in enumerate(((o, 'gray'), (w, 'gray'), (np.abs(w-o)*amp, 'magma'))):
            axs[i, j].imshow(a, cmap=cm, vmin=0, vmax=1, interpolation='nearest'); axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
        axs[0, j].set_title(labels[j], fontsize=16)
    for i, t in enumerate(row_labels): axs[i, 0].set_ylabel(t, fontsize=16)
    fig.savefig(out, dpi=100); plt.close(fig)
def _norm(a, n=256):
    import cv2
    a = a.astype(np.float32); a = (a-np.percentile(a, 0.5))/(np.percentile(a, 99.5)-np.percentile(a, 0.5)+1e-6); a = np.clip(a, 0, 1); h, w = a.shape; s = min(h, w); a = a[(h-s)//2:(h-s)//2+s, (w-s)//2:(w-s)//2+s]
    return cv2.resize(a, (n, n), interpolation=cv2.INTER_CUBIC).clip(0, 1)
def fig11_standin(out):
    import pydicom; from pydicom.data import get_testdata_file; from skimage.data import shepp_logan_phantom
    L = [('CT (pydicom CT_small, 128²)', _norm(pydicom.dcmread(get_testdata_file('CT_small.dcm')).pixel_array)), ('MR (pydicom examples_overlay, 300×484)', _norm(pydicom.dcmread(get_testdata_file('examples_overlay.dcm')).pixel_array)),
         ('MR (pydicom MR_small, 64²)', _norm(pydicom.dcmread(get_testdata_file('MR_small.dcm')).pixel_array)), ('Shepp–Logan phantom (skimage)', _norm(shepp_logan_phantom()))]
    _grid([_standin(a) for _, a in L], [t for t, _ in L], out, ['Original I', 'Stand-in I_w (≈ 40 dB)', '|I_w − I| × 50'])
def fig11_trained(out, run, mias, n=6):
    import data as D, evaluate as E
    dev = torch.device('cpu'); s, meta = E.load_run(run, dev); fac = P.PayloadFactory(meta['variant'], seed=4242); te = D.patient_split(D.mias_records(mias))['test']; ds = D.ImageSet(te[:n])
    val = torch.utils.data.DataLoader(D.ImageSet(D.patient_split(D.mias_records(mias))['val']), batch_size=8); E.calibrate_alpha(s, val, P.PayloadFactory(meta['variant'], seed=777), dev); items = []
    with torch.no_grad():
        for i in range(len(ds)):
            x = ds[i][0][None]; m, rk, _ = fac.batch(1); w = s.embed(x, torch.from_numpy(m), torch.from_numpy(rk)); items.append((x[0, 0].numpy(), w[0, 0].numpy()))
    _grid(items, [r['name'] for r in te[:n]], out, ['Original I', 'Watermarked I_w (trained encoder, 40 dB)', '|I_w − I| × 50'])
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('what', choices=['conceptual', 'fig11_standin', 'fig11', 'fig1']); ap.add_argument('out'); ap.add_argument('--run'); ap.add_argument('--mias'); ap.add_argument('--src'); ap.add_argument('--n', type=int, default=6); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.what == 'conceptual': fig3(f'{a.out}/fig3_payload.png'); fig7(f'{a.out}/fig7_verification.png')
    elif a.what == 'fig1': patch_fig1(a.src, f'{a.out}/fig1_pipeline.png')
    elif a.what == 'fig11_standin': fig11_standin(f'{a.out}/fig11_standin.png')
    else: fig11_trained(f'{a.out}/fig11_trained.png', a.run, a.mias, a.n)
