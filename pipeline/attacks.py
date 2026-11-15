"""Attack layer (Section 3.5, Table 7) and synthetic tampering for the localiser. Images: B,1,H,W in [0,1]."""
import math, io, random, numpy as np, torch, torch.nn.functional as F
from PIL import Image

def q8(x):                                   # Eq. (15): 8-bit rounding with straight-through gradient
    return x + (torch.round(255*x)/255 - x).detach()

# ---------- differentiable JPEG ----------
_QL = torch.tensor([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]], dtype=torch.float32)
def _dct_mat():
    n = 8; m = torch.zeros(n, n)
    for k in range(n):
        for i in range(n): m[k, i] = math.sqrt((1 if k else .5)*2/n)*math.cos((2*i+1)*k*math.pi/(2*n))
    return m
_D = _dct_mat()
def rnd_d(x): r = torch.round(x); return r + (x-r)**3            # Eq. (17)
def jpeg_diff(x, quality):
    B, _, H, W = x.shape; D = _D.to(x); Ql = _QL.to(x)
    s = 5000/quality if quality < 50 else 200-2*quality
    Qm = torch.clamp(torch.floor((Ql*s+50)/100), 1, 255)
    blocks = F.unfold(x*255-128, 8, stride=8).transpose(1, 2).reshape(B, -1, 8, 8)
    c = D @ blocks @ D.T
    c = rnd_d(c/Qm)*Qm
    r = (D.T @ c @ D).reshape(B, -1, 64).transpose(1, 2)
    return (F.fold(r, (H, W), 8, stride=8)+128)/255
def jpeg_real(x, quality):                    # real JPEG codec (no gradient; used mixed with the differentiable version)
    out = []
    for im in (x.detach().cpu().numpy()[:, 0]*255).round().astype(np.uint8):
        buf = io.BytesIO(); Image.fromarray(im).save(buf, 'JPEG', quality=int(quality)); out.append(np.array(Image.open(buf)))
    y = torch.from_numpy(np.stack(out)).float().unsqueeze(1).to(x)/255
    return x + (y-x).detach()                 # straight-through

# ---------- signal attacks ----------
def gauss_noise(x, sigma): return x + torch.randn_like(x)*sigma/255
def gauss_blur(x, sigma):
    k = int(2*math.ceil(3*sigma)+1); ax = torch.arange(k, device=x.device, dtype=x.dtype)-k//2
    g = torch.exp(-ax**2/(2*sigma**2)); g = g/g.sum()
    x = F.conv2d(F.pad(x, (k//2, k//2, 0, 0), mode='reflect'), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(x, (0, 0, k//2, k//2), mode='reflect'), g.view(1, 1, k, 1))
def median(x, k):
    p = k//2; u = F.unfold(F.pad(x, (p, p, p, p), mode='reflect'), k)          # B,k*k,HW
    return u.median(1).values.view_as(x)
# ---------- geometric ----------
def affine(x, angle_deg=0., scale=1., shear=0.):
    B = x.shape[0]; a = math.radians(angle_deg)
    A = torch.tensor([[math.cos(a)/scale, -math.sin(a)/scale+shear, 0], [math.sin(a)/scale, math.cos(a)/scale, 0]], dtype=x.dtype, device=x.device)
    return F.grid_sample(x, F.affine_grid(A.expand(B, 2, 3), x.shape, align_corners=False), padding_mode='zeros', align_corners=False)
def crop_zero(x, keep):                      # keep a centred rectangle with `keep` of the area, zero the rest (training attack)
    H, W = x.shape[-2:]; f = math.sqrt(keep); h, w = int(H*f), int(W*f); t, l = (H-h)//2, (W-w)//2
    m = torch.zeros_like(x); m[..., t:t+h, l:l+w] = 1; return x*m
def crop_resize(x, keep):                    # real crop-out followed by resize to the original size (extra test: shifts alignment)
    H, W = x.shape[-2:]; f = math.sqrt(keep); h, w = int(H*f), int(W*f); t, l = (H-h)//2, (W-w)//2
    return F.interpolate(x[..., t:t+h, l:l+w], size=(H, W), mode='bilinear', align_corners=False)
def packet_loss(x, frac, blk=16):
    B, _, H, W = x.shape; gh, gw = H//blk, W//blk
    m = (torch.rand(B, 1, gh, gw, device=x.device) >= frac).to(x)
    return x*F.interpolate(m, scale_factor=blk, mode='nearest')
# ---------- held-out attacks (never used in training) ----------
def speckle(x, var): return x + x*torch.randn_like(x)*math.sqrt(var)
def sharpen(x, radius=2, amount=1.0): return x + amount*(x - gauss_blur(x, radius))
def contrast(x, gain): return (x-0.5)*gain+0.5
def jp2k(x, bpp=0.5):
    out = []
    for im in (x.detach().cpu().numpy()[:, 0]*255).round().astype(np.uint8):
        buf = io.BytesIO(); Image.fromarray(im).save(buf, 'JPEG2000', quality_mode='rates', quality_layers=[8/bpp], irreversible=True)
        out.append(np.array(Image.open(buf)))
    return torch.from_numpy(np.stack(out)).float().unsqueeze(1).to(x)/255
DENOISER = None                              # set by evaluate.py --denoiser: a trained network (e.g. DnCNN, sigma = 15) used as the regeneration attack
def nlm_denoise(x, h=10):                    # default stand-in regeneration attack (non-local means)
    if DENOISER is not None:
        with torch.no_grad(): return DENOISER(x).clamp(0, 1)
    import cv2
    out = [cv2.fastNlMeansDenoising(im, None, h, 7, 21) for im in (x.detach().cpu().numpy()[:, 0]*255).round().astype(np.uint8)]
    return torch.from_numpy(np.stack(out)).float().unsqueeze(1).to(x)/255

# ---------- the training attack layer (Table 7, Eq. 16) ----------
class AttackLayer:
    def __init__(self, p_identity=0.15, p_real_jpeg=0.5): self.p_id, self.p_real = p_identity, p_real_jpeg
    def __call__(self, x, s=1.0):
        """s in [0,1]: curriculum strength (Eq. 22). One attack drawn per sample."""
        out = x.clone(); R = random.Random(); names = []
        for i in range(x.shape[0]):
            xi = x[i:i+1]
            if R.random() < self.p_id: names.append('identity'); continue
            a = R.choice(['noise', 'blur', 'median', 'jpeg', 'rot', 'scale', 'crop', 'packet'])
            if a == 'noise': xi = gauss_noise(xi, R.uniform(0, 10*s))
            elif a == 'blur': xi = gauss_blur(xi, 0.5+R.uniform(0, 1.5*s)) if s > 0 else xi
            elif a == 'median': xi = median(xi, R.choice([3, 5])) if s > 0 else xi
            elif a == 'jpeg':
                qq = 90 - R.uniform(0, 40*s); xi = jpeg_real(xi, qq) if R.random() < self.p_real else jpeg_diff(xi, qq)
            elif a == 'rot': xi = affine(xi, angle_deg=R.uniform(-15*s, 15*s), scale=1+R.uniform(-.2*s, .2*s), shear=R.uniform(-.1*s, .1*s))
            elif a == 'scale': xi = affine(xi, scale=1+R.uniform(-.2*s, .2*s))
            elif a == 'crop': xi = crop_zero(xi, 1-R.uniform(0, .25*s))
            elif a == 'packet': xi = packet_loss(xi, R.uniform(.01, .01+.04*s))
            out[i:i+1] = xi; names.append(a)
        return q8(out.clamp(0, 1)), names

# ---------- synthetic tampering for the localiser ----------
def tamper(x, other=None, area=(0.01, 0.15), kinds=('copy', 'splice', 'blur', 'fill', 'contrast')):
    """x: B,1,H,W. Returns tampered copy and exact mask (B,1,H,W). Types: copy-move, splice, blur/fill, contrast."""
    B, _, H, W = x.shape; y = x.clone(); mask = torch.zeros_like(x); R = random.Random()
    for i in range(B):
        a = R.uniform(*area)*H*W; h = max(8, int(math.sqrt(a*R.uniform(.6, 1.6)))); w = max(8, int(a/h)); h, w = min(h, H-2), min(w, W-2)
        t, l = R.randint(0, H-h), R.randint(0, W-w); kind = R.choice(list(kinds))
        if kind == 'copy':
            t2, l2 = R.randint(0, H-h), R.randint(0, W-w); y[i, :, t:t+h, l:l+w] = x[i, :, t2:t2+h, l2:l2+w]
        elif kind == 'splice' and other is not None:
            t2, l2 = R.randint(0, H-h), R.randint(0, W-w); y[i, :, t:t+h, l:l+w] = other[i, :, t2:t2+h, l2:l2+w]
        elif kind == 'blur': y[i:i+1, :, t:t+h, l:l+w] = gauss_blur(x[i:i+1, :, t:t+h, l:l+w], 2.0)
        elif kind == 'contrast': y[i, :, t:t+h, l:l+w] = ((x[i, :, t:t+h, l:l+w]-.5)*R.choice([.6, 1.5])+.5).clamp(0, 1)
        else: y[i, :, t:t+h, l:l+w] = x[i, :, t:t+h, l:l+w].mean()
        mask[i, :, t:t+h, l:l+w] = 1
    return q8(y.clamp(0, 1)), mask
