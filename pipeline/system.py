"""Proposed system wrapper + pruning (network slimming) + fake-quantisation (QAT) utilities."""
import copy, torch, torch.nn as nn, torch.nn.functional as F
from models import *
from attacks import q8

class SpatialEncoder(Encoder):
    """Ablation (v): same network, but operating on the 256x256 image directly (no DWT)."""
    def forward(self, img, m, rk):
        D = F.avg_pool2d((img-F.avg_pool2d(img, 3, 1, 1))**2, 5, 1, 2); D = D/(D.amax((2, 3), keepdim=True)+1e-8)
        up = lambda t: F.interpolate(t, size=img.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([img, D, up(self.message_map(m, 128)), up(rk)], 1)
        return img + self.alpha*self.head(self.blocks(self.stem(x)))

class System(nn.Module):
    def __init__(self, cfg):
        super().__init__(); s = cfg
        self.cfg = cfg
        self.enc = (SpatialEncoder if s.get('spatial_encoder') else Encoder)(C=s.get('C', 24))
        self.dec = Decoder(use_stn=not s.get('no_stn'), global_pool=s.get('global_pool'))
        self.loc = Localiser()
        self.enc.no_d = bool(s.get('no_d'))
    def embed(self, img, m, rk):
        if self.cfg.get('no_rk'): rk = torch.zeros_like(rk)
        return q8(self.enc(img, m, rk).clamp(0, 1))
    def extract(self, att): return self.dec(att)[0]
    def localise(self, img, rk):
        if self.cfg.get('no_rk'): rk = torch.zeros_like(rk)
        return self.loc(img, rk)

# ---------------- network slimming + channel pruning of the encoder body ----------------
def bn_gammas(enc): return [enc.stem[1].weight]+[b.bn1.weight for b in enc.blocks]+[b.bn2.weight for b in enc.blocks]
def slim_penalty(enc): return sum(g.abs().sum() for g in bn_gammas(enc))
@torch.no_grad()
def prune_encoder(enc, ratio):
    C = enc.stem[0].out_channels; keep_n = max(4, round(C*(1-ratio)))
    score = sum(g.abs() for g in bn_gammas(enc)); keep = score.topk(keep_n).indices.sort().values
    new = type(enc)(C=keep_n, alpha=enc.alpha); new.msg.load_state_dict(enc.msg.state_dict()); new.no_d = getattr(enc, 'no_d', False)
    def cp_bn(d, s): [getattr(d, k).copy_(getattr(s, k)[keep]) for k in ('weight', 'bias', 'running_mean', 'running_var')]
    new.stem[0].weight.copy_(enc.stem[0].weight[keep]); cp_bn(new.stem[1], enc.stem[1])
    for nb, ob in zip(new.blocks, enc.blocks):
        nb.dw.weight.copy_(ob.dw.weight[keep]); cp_bn(nb.bn1, ob.bn1); nb.pw.weight.copy_(ob.pw.weight[keep][:, keep]); cp_bn(nb.bn2, ob.bn2)
    new.head.weight.copy_(enc.head.weight[:, keep]); new.head.bias.copy_(enc.head.bias)
    return new, 1-keep_n/C

# ---------------- fake quantisation (Eq. 25), simulated INT8 ----------------
def _ste_round(x): return x + (torch.round(x)-x).detach()
class FQAct(nn.Module):
    def __init__(s, momentum=0.01): super().__init__(); s.register_buffer('mn', torch.zeros(1)); s.register_buffer('mx', torch.zeros(1)); s.register_buffer('init', torch.zeros(1)); s.m = momentum; s.enabled = True
    def forward(s, x):
        if not s.enabled: return x
        if s.training or s.init.item() == 0:
            mn, mx = x.detach().min(), x.detach().max()
            if s.init.item() == 0: s.mn.copy_(mn); s.mx.copy_(mx); s.init.fill_(1)
            else: s.mn.mul_(1-s.m).add_(s.m*mn); s.mx.mul_(1-s.m).add_(s.m*mx)
        sc = (s.mx-s.mn).clamp_min(1e-8)/255; z0 = -128-torch.round(s.mn/sc)
        return sc*(torch.clamp(_ste_round(x/sc)+z0, -128, 127)-z0)
def fq_weight(w):
    sc = w.detach().abs().max().clamp_min(1e-8)/127
    return sc*torch.clamp(_ste_round(w/sc), -127, 127)
class QConv(nn.Module):
    def __init__(s, c): super().__init__(); s.c = c; s.q = FQAct()
    def forward(s, x):
        c = s.c; return F.conv2d(s.q(x), fq_weight(c.weight), c.bias, c.stride, c.padding, c.dilation, c.groups)
class QLinear(nn.Module):
    def __init__(s, l): super().__init__(); s.l = l; s.q = FQAct()
    def forward(s, x): return F.linear(s.q(x), fq_weight(s.l.weight), s.l.bias)
def to_qat(module):
    for n, ch in list(module.named_children()):
        if isinstance(ch, nn.Conv2d): setattr(module, n, QConv(ch))
        elif isinstance(ch, nn.Linear): setattr(module, n, QLinear(ch))
        else: to_qat(ch)
    return module
def model_size_bytes(m, int8=False):
    n = sum(p.numel() for p in m.parameters()); return n*(1 if int8 else 4)
