import math, random, numpy as np, torch, torch.nn.functional as F
from pytorch_msssim import ms_ssim

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def psnr(a, b, mask=None):                       # MAX = 255 on rounded 8-bit images; per-sample
    a, b = (a*255).round(), (b*255).round(); d = (a-b)**2
    mse = d.flatten(1).mean(1) if mask is None else (d*mask).flatten(1).sum(1)/mask.flatten(1).sum(1).clamp_min(1)
    return 10*torch.log10(255**2/mse.clamp_min(1e-10))
def ssim(a, b): return ms_ssim(a, b, data_range=1.0, size_average=False)
def dice(p, t, eps=1e-6):
    i = (p*t).flatten(1).sum(1); return (2*i+eps)/(p.flatten(1).sum(1)+t.flatten(1).sum(1)+eps)
def dice_loss(p, t): return (1-dice(p, t)).mean()
def to_mask128(m): return F.adaptive_avg_pool2d(m, 128)
def get_device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def img_loss(img, iw, roi, omega=5.0, beta=0.1):          # Eq. (19)
    w = 1+(omega-1)*roi
    return (w*(img-iw)**2).flatten(1).sum(1).div(w.flatten(1).sum(1)).mean() + beta*(1-ms_ssim(img, iw, data_range=1.0)).mean()
