"""Payload chain: metadata -> SHA-256 -> AES-128-CBC (or HMAC) -> BCH(511,385,t=14) -> keyed permutation."""
import numpy as np
import pytest

import payload as P

KEY, IV, META = bytes(range(16)), bytes(range(16, 32)), b"patient-0001|MR|2026-01-01"
VARIANTS = ["cbc", "hmac"]


def flip(cw, n, seed=1):
    r = cw.copy()
    r[np.random.default_rng(seed).choice(P.N, n, replace=False)] ^= 1
    return r


@pytest.mark.parametrize("variant", VARIANTS)
def test_round_trip_is_authentic(variant):
    cw = P.encode(KEY, META, IV, variant)
    assert cw.shape == (P.N,) and set(np.unique(cw)) <= {0, 1}
    assert P.verify(cw, KEY, META, variant) == "authentic"


@pytest.mark.parametrize("variant", VARIANTS)
def test_wrong_key_is_never_authentic(variant):
    cw = P.encode(KEY, META, IV, variant)
    for other in (bytes(16), bytes([1]) + bytes(range(1, 16))):      # unrelated key, and a 1-byte change
        assert P.verify(cw, other, META, variant) in {"unreadable", "metadata mismatch"}


@pytest.mark.parametrize("variant", VARIANTS)
def test_other_metadata_is_a_mismatch(variant):
    cw = P.encode(KEY, META, IV, variant)
    assert P.verify(cw, KEY, b"patient-0002|MR|2026-01-01", variant) == "metadata mismatch"


@pytest.mark.parametrize("variant", VARIANTS)
def test_bch_corrects_up_to_t_errors_and_not_beyond(variant):
    cw = P.encode(KEY, META, IV, variant)
    assert P.T == 14
    assert P.verify(flip(cw, P.T), KEY, META, variant) == "authentic"            # t = 14 errors are corrected
    assert P.verify(flip(cw, 40), KEY, META, variant) != "authentic"             # far beyond t is never accepted


def test_permutation_is_a_key_dependent_permutation():
    p1, p2 = P.permutation(P.derive(KEY)["perm"]), P.permutation(P.derive(bytes(16))["perm"])
    assert sorted(p1) == list(range(P.N))
    assert (P.permutation(P.derive(KEY)["perm"]) == p1).all()                    # deterministic
    assert (p1 != p2).any()                                                      # and depends on the key


def test_derived_keys_are_separated():
    d = P.derive(KEY)
    assert len({d["enc"], d["mac"], d["perm"], d["rk"]}) == 4 and all(len(v) == 16 for v in d.values())


def test_reference_pattern_is_pm1_deterministic_and_key_dependent():
    r1 = P.reference_pattern(P.derive(KEY)["rk"], IV)
    assert r1.shape == (128, 128) and set(np.unique(r1)) == {-1.0, 1.0}
    assert (r1 == P.reference_pattern(P.derive(KEY)["rk"], IV)).all()
    agree = (r1 == P.reference_pattern(P.derive(bytes(16))["rk"], IV)).mean()
    assert 0.45 < agree < 0.55                                                   # an unrelated key gives an unrelated pattern


@pytest.mark.parametrize("variant", VARIANTS)
def test_factory_batch_is_decodable(variant):
    msgs, rks, aux = P.PayloadFactory(variant=variant, seed=3).batch(4)
    assert msgs.shape == (4, P.N) and rks.shape == (4, 1, 128, 128)
    for m, (key, md, iv) in zip(msgs, aux):
        assert P.verify(m.astype(np.uint8), key, md, variant) == "authentic"
    assert len({a[0] for a in aux}) == 4                                         # a fresh key per sample
