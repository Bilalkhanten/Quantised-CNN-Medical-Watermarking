"""PSNR/SSIM of each test attack on the four stand-in watermarked images of Fig. 11 (mean +- SD, n=4). Writes table10_measured.json"""
import json, numpy as np, torch, pydicom
from pydicom.data import get_testdata_file
from skimage.data import shepp_logan_phantom
from skimage.metrics import structural_similarity as ssim
import attacks as A
from make_figures import _norm, _standin
src = [pydicom.dcmread(get_testdata_file(f)).pixel_array for f in ('CT_small.dcm', 'examples_overlay.dcm', 'MR_small.dcm')] + [shepp_logan_phantom()]
Ws = [_standin(_norm(a, 256))[1] for a in src]; T = lambda a: torch.from_numpy(a).float()[None, None]
att = {('JPEG', 'Q = 70'): lambda x: A.jpeg_real(x, 70), ('JPEG', 'Q = 50'): lambda x: A.jpeg_real(x, 50), ('Gaussian noise', 'σ = 10'): lambda x: A.gauss_noise(x, 10),
 ('Median filter', '3 × 3'): lambda x: A.median(x, 3), ('Median filter', '5 × 5'): lambda x: A.median(x, 5), ('Gaussian blur', 'σ = 2'): lambda x: A.gauss_blur(x, 2),
 ('Rotation', '15°'): lambda x: A.affine(x, 15.), ('Scaling', '0.8'): lambda x: A.affine(x, 0., .8), ('Cropping', '75% kept'): lambda x: A.crop_zero(x, .75), ('Packet loss', '5% of blocks'): lambda x: A.packet_loss(x, .05),
 ('Speckle noise', 'σ² = 0.04'): lambda x: A.speckle(x, .04), ('Sharpening', 'unsharp mask, r = 2'): lambda x: A.sharpen(x, 2), ('Contrast change', 'gain × 1.3'): lambda x: A.contrast(x, 1.3),
 ('JPEG 2000', '0.5 bpp'): lambda x: A.jp2k(x, .5), ('Denoiser (NLM stand-in)', 'non-local means, h = 10'): lambda x: A.nlm_denoise(x, 10)}
out = {}
for k, f in att.items():
    torch.manual_seed(0); ps, ss = [], []
    for w in Ws:
        y = (f(T(w)).clamp(0, 1)*255).round()/255; y = y[0, 0].numpy()
        ps.append(10*np.log10(1/max(((y-w)**2).mean(), 1e-12))); ss.append(ssim(w, y, data_range=1.0))
    out['|'.join(k)] = [float(np.mean(ps)), float(np.std(ps, ddof=1)), float(np.mean(ss)), float(np.std(ss, ddof=1))]
    print(k, '%.1f ± %.1f  %.3f ± %.3f' % tuple(out['|'.join(k)]))
json.dump(out, open('table10_measured.json', 'w'), indent=1)
