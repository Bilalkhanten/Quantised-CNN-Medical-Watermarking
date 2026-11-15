"""Architecture sizes (Table 6), Haar transform, pruning and the attack layer."""
import pytest
import torch

import attacks as A
import models as M
from system import prune_encoder

torch.manual_seed(0)


def _inputs():
    x = torch.rand(1, 1, 256, 256)
    m = torch.randint(0, 2, (1, 511)).float()
    rk = torch.randint(0, 2, (1, 1, 128, 128)).float() * 2 - 1
    return x, m, rk


def test_parameter_counts_reproduce_table_6():
    assert M.count_params(M.Encoder()) == 529_425
    assert M.count_params(M.Decoder()) == 587_589
    assert M.count_params(M.Localiser()) == 51_489


def test_mac_counts_reproduce_table_6():
    x, m, rk = _inputs()
    assert round(M.count_macs(M.Encoder(), x, m, rk) / 1e6, 2) == 77.59
    assert round(M.count_macs(M.Decoder(), x) / 1e6, 2) == 105.60
    assert round(M.count_macs(M.Localiser(), x, rk) / 1e6, 2) == 299.89


def test_forward_shapes():
    x, m, rk = _inputs()
    enc, dec, loc = M.Encoder().eval(), M.Decoder().eval(), M.Localiser().eval()
    with torch.no_grad():
        w = enc(x, m, rk)
        assert w.shape == x.shape
        logits, theta, rectified = dec(w)                     # 511 bit logits, spatial-transformer parameters, rectified image
        assert logits.shape == (1, 511) and theta.shape == (1, 2, 3) and rectified.shape == x.shape
        mask = loc(w, rk)                                     # tamper map in the half-resolution (Haar LL) domain
        assert mask.shape == (1, 1, 128, 128) and 0 <= mask.min() and mask.max() <= 1


def test_pruning_at_0_38_keeps_15_channels():
    enc = M.Encoder()
    pruned, ratio = prune_encoder(enc, 0.38)
    assert pruned.stem[0].out_channels == 15
    assert M.count_params(pruned) == 526_959
    assert ratio == pytest.approx(1 - 15 / 24)


def test_haar_is_orthonormal_and_invertible():
    x = torch.rand(2, 1, 32, 32)
    y = M.haar_dwt(x)
    assert y.shape == (2, 4, 16, 16)
    assert torch.allclose(M.haar_idwt(y), x, atol=1e-6)
    assert torch.allclose((y ** 2).sum(), (x ** 2).sum(), rtol=1e-5)             # energy is preserved


ATTACKS = {
    "jpeg_real": lambda x: A.jpeg_real(x, 70), "jpeg_diff": lambda x: A.jpeg_diff(x, 70), "gauss_noise": lambda x: A.gauss_noise(x, 10),
    "gauss_blur": lambda x: A.gauss_blur(x, 2), "median": lambda x: A.median(x, 3), "affine": lambda x: A.affine(x, 15.),
    "crop_zero": lambda x: A.crop_zero(x, 0.75), "crop_resize": lambda x: A.crop_resize(x, 0.75), "packet_loss": lambda x: A.packet_loss(x, 0.05),
    "speckle": lambda x: A.speckle(x, 0.04), "sharpen": lambda x: A.sharpen(x, 2), "contrast": lambda x: A.contrast(x, 1.3),
    "jp2k": lambda x: A.jp2k(x, 0.5), "nlm_denoise": lambda x: A.nlm_denoise(x, 10),
}


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_attack_keeps_shape_and_is_finite(name):
    x = torch.rand(1, 1, 128, 128)
    y = ATTACKS[name](x)
    assert y.shape == x.shape and torch.isfinite(y).all()


def test_q8_rounds_to_8_bit_levels_with_straight_through_gradient():
    x = torch.rand(1, 1, 8, 8, requires_grad=True)
    y = A.q8(x)
    assert torch.allclose(y * 255, (y * 255).round(), atol=1e-4)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))
