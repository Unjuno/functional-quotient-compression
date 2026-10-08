from pathlib import Path
import sys

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SRC))
from engine import decode, make_state, make_world, save_payload, load_payload


def test_support_and_heldout_are_disjoint_and_cover_cross_product():
    _, _, _, _, support = make_world(26601, "aligned")
    assert int(support.sum()) == 8
    assert np.all(support.sum(axis=0) > 0)
    assert np.all(support.sum(axis=1) > 0)
    heldout = ~support
    assert not np.any(support & heldout)
    assert np.all(support | heldout)
    assert int(heldout.sum()) == 8


def test_mirror_and_coefficient_product_are_same_aligned_function():
    core, teachers, _, _, _ = make_world(26601, "aligned")
    mirror = make_state("mirror", core, teachers, 26601)
    coeff = make_state("coeff_product", core, teachers, 26601)
    for i in range(4):
        for j in range(4):
            np.testing.assert_allclose(decode("mirror", mirror, i, j),
                                       decode("coeff_product", coeff, i, j), atol=5e-4, rtol=5e-4)


def test_mirror_recovers_aligned_teacher_to_frozen_quality_scale():
    core, teachers, _, _, _ = make_world(26602, "aligned")
    state = make_state("mirror", core, teachers, 26602)
    err = np.mean((np.stack([[decode("mirror", state, i, j) for j in range(4)] for i in range(4)]) - teachers) ** 2)
    assert err < 1e-6


def test_payload_is_byte_stable_and_roundtrips(tmp_path):
    core, teachers, _, _, _ = make_world(26603, "aligned")
    state = make_state("coeff_product", core, teachers, 26603)
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    assert save_payload(a, state) == save_payload(b, state)
    assert a.read_bytes() == b.read_bytes()
    got = load_payload(a)
    for k in state:
        np.testing.assert_array_equal(got[k], state[k])
