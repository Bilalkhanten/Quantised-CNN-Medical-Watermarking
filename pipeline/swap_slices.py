"""Regenerate/patch the slice-bearing figures with a traceable image (pydicom examples_overlay MR slice).
usage: python swap_slices.py MEDIA_DIR OUT_DIR   (MEDIA_DIR = original word/media; OUT_DIR gets image4,6,7,8,12)"""
import sys, os, numpy as np, torch, torch.nn.functional as F, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.cm as cm, pydicom
from pydicom.data import get_testdata_file
from PIL import Image, ImageDraw
from models import haar_dwt, haar_idwt, detail_energy
import attacks as A
from make_figures import _norm, _standin
MED, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
I, W = _standin(_norm(pydicom.dcmread(get_testdata_file('examples_overlay.dcm')).pixel_array, 256))   # 256x256 in [0,1]; W = stand-in watermarked
T = lambda a: torch.from_numpy(a).float()[None, None]
y = haar_dwt(T(I)); D = detail_energy(y)[0, 0].numpy()
def u8(a): return (np.clip(a, 0, 1)*255).round().astype(np.uint8)
def gray(a): return Image.fromarray(u8(a), 'L').convert('RGB')
def cmap(a, name, lo=None, hi=None):
    lo = a.min() if lo is None else lo; hi = a.max() if hi is None else hi
    return Image.fromarray((getattr(cm, name)(np.clip((a-lo)/(hi-lo+1e-9), 0, 1))[..., :3]*255).astype(np.uint8))
def paste(im, box, tile):
    x, yy, w, h = box; im.paste(tile.resize((w, h), Image.LANCZOS if tile.size[0] > w else Image.BICUBIC), (x, yy))
    ImageDraw.Draw(im).rectangle([x, yy, x+w-1, yy+h-1], outline=(40, 40, 40), width=2)
def load(n): return Image.open(f'{MED}/image{n}.png').convert('RGB')
def ll_img(a): l = haar_dwt(T(a))[0, 0].numpy()/2; return gray(l)
def dmap(a):
    d = detail_energy(haar_dwt(T(a)))[0, 0].numpy(); return cmap(d, 'magma', 0, np.percentile(d, 99.5))
# ---------- Fig 4(a)  (image6) ----------
im = load(6); res = np.abs(W-I)
for x, tile in [(22, gray(I)), (292, ll_img(I)), (524, dmap(I)), (1469, gray(W)), (1709, cmap(res*50, 'plasma', 0, 1))]:
    paste(im, (x, 120, 213, 213), tile)
im.save(f'{OUT}/image6.png')
# ---------- tampered / rotated variants ----------
t0, l0, h0, w0 = 148, 151, 40, 48; Tm = W.copy(); Tm[t0:t0+h0, l0:l0+w0] = W[128:128+h0, 72:72+w0]
rot = A.jpeg_real(A.affine(T(W), 8.), 70)[0, 0].numpy()
# ---------- Fig 5(c)  (image7) ----------
im = load(7)
for x, tile in [(22, gray(rot)), (299, gray(W)), (599, gray(Tm)), (839, ll_img(Tm)), (1072, dmap(Tm))]:
    paste(im, (x, 1679, 205, 205), tile)
im.save(f'{OUT}/image7.png')
# ---------- Fig 6  (image8) ----------
im = load(8)
for box, tile in [((17, 353, 101, 101), gray(I)), ((292, 353, 100, 101), gray(W)), ((1252, 420, 100, 102), gray(Tm))]: paste(im, box, tile)
im.save(f'{OUT}/image8.png')
# ---------- Fig 2  (image4): rebuilt ----------
bands = haar_dwt(T(I))[0].numpy(); fig = plt.figure(figsize=(19.8, 9.45), dpi=100); fig.patch.set_facecolor('white'); S = 19.8
def ax_(l, b, w, h): a = fig.add_axes([l/1980, 1-(b+h)/945, w/1980, h/945]); a.set_xticks([]); a.set_yticks([]); return a
def band(a, v): a.imshow(v, cmap='gray', vmin=-np.percentile(np.abs(v), 99.5), vmax=np.percentile(np.abs(v), 99.5))
a = ax_(20, 222, 529, 529); a.imshow(I, cmap='gray', vmin=0, vmax=1); fig.text(284/1980, 1-186/945, '(a) Image I, 256 × 256', ha='center', va='center', fontsize=21)
a = ax_(572, 98, 330, 330); a.imshow(bands[0], cmap='gray'); [s.set_color('#c0392b') or s.set_linewidth(4) for s in a.spines.values()]; fig.text(737/1980, 1-62/945, '(b) LL, 128 × 128', ha='center', va='center', fontsize=21)
for x, k, t in [(924, 1, '(c) LH'), (1277, 2, '(d) HL'), (1630, 3, '(e) HH')]:
    a = ax_(x, 98, 330, 330); band(a, bands[k]); fig.text((x+165)/1980, 1-62/945, t, ha='center', va='center', fontsize=21)
a = ax_(1097, 541, 338, 338); a.imshow(D, cmap='magma'); fig.text(1265/1980, 1-506/945, '(f) Detail-energy map D = |LH| + |HL| + |HH|', ha='center', va='center', fontsize=21)
fig.text(572/1980, 1-700/945, 'Embedding band\n(red frame):\nLL carries the\npayload and R$_K$', ha='left', va='center', fontsize=21, color='#c0392b', linespacing=1.3)
fig.text(1630/1980, 1-708/945, 'D tells the encoder\nwhere texture can\nmask a change', ha='left', va='center', fontsize=21, linespacing=1.3)
fig.savefig(f'{OUT}/image4.png', dpi=100); plt.close(fig)
# ---------- Fig 10  (image12): real attacks, PSNR computed ----------
Wt = T(W); torch.manual_seed(0)
P = lambda z: 10*np.log10(1/max(float(((z-Wt)**2).mean()), 1e-12))
cases = [('Watermarked I$_w$\n(stand-in)', Wt), ('JPEG\nQ = 90', A.jpeg_real(Wt, 90)), ('JPEG\nQ = 70', A.jpeg_real(Wt, 70)), ('JPEG\nQ = 50', A.jpeg_real(Wt, 50)),
 ('Gaussian noise\nσ = 3', A.gauss_noise(Wt, 3).clamp(0, 1)), ('Gaussian noise\nσ = 10', A.gauss_noise(Wt, 10).clamp(0, 1)), ('Median filter\n3 × 3', A.median(Wt, 3)),
 ('Median filter\n5 × 5', A.median(Wt, 5)), ('Gaussian blur\nσ = 2', A.gauss_blur(Wt, 2)), ('Rotation\n15°', A.affine(Wt, 15.)), ('Scaling\n0.8', A.affine(Wt, 0., .8)),
 ('Cropping\n75% kept', A.crop_zero(Wt, .75)), ('Packet loss\n5% of blocks', A.packet_loss(Wt, .05))]
cases = cases[:12] + [cases[12]] if False else cases
fig = plt.figure(figsize=(19.8, 8.55), dpi=100)
pos = [(20+i*279, 122) for i in range(7)] + [(20+i*279, 586) for i in range(6)]
for (t, z), (x, yy) in zip(cases, pos):
    a = fig.add_axes([x/1980, 1-(yy+266)/855, 266/1980, 266/855]); a.imshow(z[0, 0].numpy(), cmap='gray', vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
    lab = t if t.startswith('Watermarked') else f'{t}\n{P(z):.1f} dB'; fig.text((x+133)/1980, 1-(yy-6)/855, lab, ha='center', va='bottom', fontsize=16, linespacing=1.1)
fig.savefig(f'{OUT}/image12.png', dpi=100); plt.close(fig)
for n in (4, 6, 7, 8, 12): print(n, Image.open(f'{OUT}/image{n}.png').size, 'orig', Image.open(f'{MED}/image{n}.png').size)
