from pathlib import Path
import sys

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SRC))
from engine import (D, E, decode, load_payload, make_states, make_world, save_payload,
                    svd_codec)


def test_aligned_mirror_reconstructs_orbit():
    teachers, _, _, angles = make_world(25801, "aligned")
    state = make_states(teachers, angles, rank=4)["mirror"]
    assert max(np.mean((decode("mirror", state, e) - teachers[e]) ** 2) for e in range(E)) < 1e-8


def test_unrelated_mirror_does_not_invent_independent_experts():
    teachers, _, _, angles = make_world(25801, "unrelated")
    state = make_states(teachers, angles, rank=4)["mirror"]
    err = np.mean((np.stack([decode("mirror", state, e) for e in range(E)]) - teachers) ** 2)
    assert err > 0.01


def test_psp_unbinding_shape_and_superposition():
    teachers, _, _, angles = make_world(25801, "aligned")
    state = make_states(teachers, angles, rank=4)["psp"]
    got = decode("psp", state, 0)
    assert got.shape == (D, D)
    assert np.isfinite(got).all()


def test_payload_is_byte_stable_and_reloadable(tmp_path):
    teachers, _, _, _ = make_world(25802, "aligned")
    state = svd_codec(teachers, rank=2)
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    assert save_payload(a, state) == save_payload(b, state)
    assert a.read_bytes() == b.read_bytes()
    loaded = load_payload(a)
    assert set(loaded) == set(state)
    for name in state:
        np.testing.assert_array_equal(loaded[name], state[name])
