"""Encoder / decoder (with spatial transformer) / localiser, as specified in the paper (Section 3.4).
Parameter and MAC counts reproduce Table 6: 529,425 / 587,589 / 51,489 parameters."""
import torch, torch.nn as nn, torch.nn.functional as F

# ---------------- Haar DWT / IDWT (orthonormal, single level) ----------------
def haar_dwt(x):                       # x: B,1,H,W  ->  B,4,H/2,W/2  (LL,LH,HL,HH)
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return torch.cat([(a+b+c+d)/2, (a-b+c-d)/2, (a+b-c-d)/2, (a-b-c+d)/2], 1)
def haar_idwt(y):                      # B,4,h,w -> B,1,2h,2w
    ll, lh, hl, hh = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
    B, _, h, w = ll.shape
    out = ll.new_zeros(B, 1, 2*h, 2*w)
    out[..., 0::2, 0::2] = (ll+lh+hl+hh)/2; out[..., 0::2, 1::2] = (ll-lh+hl-hh)/2
    out[..., 1::2, 0::2] = (ll+lh-hl-hh)/2; out[..., 1::2, 1::2] = (ll-lh-hl+hh)/2
    return out
def detail_energy(y, k=5):             # D: local detail energy of the three detail bands (B,1,h,w), scaled to [0,1]
    e = (y[:, 1:]**2).sum(1, keepdim=True)
    e = F.avg_pool2d(e, k, 1, k//2)
    return e / (e.amax((2, 3), keepdim=True) + 1e-8)

class DSBlock(nn.Module):              # depthwise 3x3 + BN + ReLU, pointwise 1x1 + BN (+ReLU), no conv bias
    def __init__(s, cin, cout, stride=1, residual=False):
        super().__init__()
        s.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False); s.bn1 = nn.BatchNorm2d(cin)
        s.pw = nn.Conv2d(cin, cout, 1, bias=False); s.bn2 = nn.BatchNorm2d(cout); s.res = residual
    def forward(s, x):
        y = s.bn2(s.pw(F.relu(s.bn1(s.dw(x)))))
        return F.relu(x + y) if s.res else F.relu(y)

class Encoder(nn.Module):
    """7 input channels: LL, D, 4-ch message map, R_K.  Output: residual dLL (1ch)."""
    def __init__(s, n_bits=511, C=24, alpha=1.0, n_blocks=4):
        super().__init__()
        s.msg = nn.Linear(n_bits, 1024)                                   # 524,288 params (w+b)
        s.stem = nn.Sequential(nn.Conv2d(7, C, 3, 1, 1, bias=False), nn.BatchNorm2d(C), nn.ReLU())
        s.blocks = nn.Sequential(*[DSBlock(C, C, residual=True) for _ in range(n_blocks)])
        s.head = nn.Conv2d(C, 1, 1); nn.init.normal_(s.head.weight, 0, 1e-3); nn.init.zeros_(s.head.bias)
        s.alpha = alpha; s.no_d = False                                   # no_d: ablation (xiii), detail-energy channel zeroed
    def message_map(s, m, size=128):
        mm = s.msg(m*2-1).view(-1, 4, 16, 16)
        return F.interpolate(mm, size=(size, size), mode='bilinear', align_corners=False)
    def forward(s, img, m, rk):
        """img: B,1,256,256 in [0,1]; m: B,511 {0,1}; rk: B,1,128,128 in {-1,+1}. Returns watermarked image (unquantised)."""
        y = haar_dwt(img); ll, D = y[:, 0:1], detail_energy(y)
        if s.no_d: D = torch.zeros_like(D)
        x = torch.cat([ll, D, s.message_map(m, ll.shape[-1]), rk], 1)
        dll = s.head(s.blocks(s.stem(x)))
        y2 = torch.cat([ll + s.alpha*dll, y[:, 1:]], 1)
        return haar_idwt(y2)

class STN(nn.Module):                  # rectifier: three stride-2 convs (8,16,32; with bias) on a 64x64 copy + linear->6 (18,182 params)
    def __init__(s):
        super().__init__()
        s.c1 = nn.Conv2d(1, 8, 3, 2, 1); s.c2 = nn.Conv2d(8, 16, 3, 2, 1); s.c3 = nn.Conv2d(16, 32, 3, 2, 1)
        s.fc = nn.Linear(2048, 6)
        nn.init.zeros_(s.fc.weight); s.fc.bias.data = torch.tensor([1., 0, 0, 0, 1., 0])   # identity start
    def forward(s, x):
        z = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        z = F.relu(s.c3(F.relu(s.c2(F.relu(s.c1(z))))))
        theta = s.fc(z.flatten(1)).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False), theta

class Decoder(nn.Module):
    def __init__(s, n_bits=511, use_stn=True, global_pool=False):
        super().__init__()
        s.stn = STN() if use_stn else None; s.global_pool = global_pool
        s.stem = nn.Sequential(nn.Conv2d(4, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        s.blocks = nn.Sequential(DSBlock(32, 32, 1, True), DSBlock(32, 64, 2), DSBlock(64, 64, 1, True),
                                 DSBlock(64, 128, 2), DSBlock(128, 128, 1, True))
        s.conv = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        s.fc = nn.Linear(64 if global_pool else 1024, n_bits)
    def forward(s, img):
        theta = None
        if s.stn is not None: img, theta = s.stn(img)
        x = s.conv(s.blocks(s.stem(haar_dwt(img))))
        x = F.adaptive_avg_pool2d(x, 1 if s.global_pool else 4)
        return s.fc(x.flatten(1)), theta, img               # logits, theta, rectified image

def cbr(cin, cout): return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())
class Localiser(nn.Module):            # two-level U-Net, 16-32 channels; inputs LL', D', R_K
    def __init__(s):
        super().__init__()
        s.e1 = nn.Sequential(cbr(3, 16), cbr(16, 16)); s.e2 = nn.Sequential(cbr(16, 32), cbr(32, 32))
        s.bott = cbr(32, 32); s.d2 = cbr(64, 32); s.d1 = cbr(48, 16); s.head = nn.Conv2d(16, 1, 1)
    def forward(s, img, rk):
        y = haar_dwt(img); x = torch.cat([y[:, 0:1], detail_energy(y), rk], 1)
        e1 = s.e1(x); e2 = s.e2(F.max_pool2d(e1, 2)); b = s.bott(F.max_pool2d(e2, 2))
        u2 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = s.d2(torch.cat([u2, e2], 1))
        u1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        return torch.sigmoid(s.head(s.d1(torch.cat([u1, e1], 1))))

# ---------------- counting ----------------
def count_params(m): return sum(p.numel() for p in m.parameters())
def count_macs(model, *inputs):
    macs = [0]
    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            macs[0] += out.numel()//out.shape[0] * (mod.in_channels//mod.groups) * mod.kernel_size[0]*mod.kernel_size[1]
        elif isinstance(mod, nn.Linear): macs[0] += mod.in_features*mod.out_features
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad(): model(*inputs)
    for h in hs: h.remove()
    return macs[0]

if __name__ == '__main__':
    E, D, L = Encoder(), Decoder(), Localiser()
    x = torch.rand(1, 1, 256, 256); m = torch.randint(0, 2, (1, 511)).float(); rk = torch.randint(0, 2, (1, 1, 128, 128)).float()*2-1
    print('Encoder   params', count_params(E), 'MMACs %.2f' % (count_macs(E, x, m, rk)/1e6), ' paper: 529,425 / 77.6')
    print('Decoder   params', count_params(D), 'MMACs %.2f' % (count_macs(D, x)/1e6),        ' paper: 587,589 / 105.6')
    print('Localiser params', count_params(L), 'MMACs %.2f' % (count_macs(L, x, rk)/1e6),    ' paper: 51,489 / 299.9')
