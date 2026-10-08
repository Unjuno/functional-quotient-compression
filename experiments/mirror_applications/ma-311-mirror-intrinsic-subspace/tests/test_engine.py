from pathlib import Path
import sys

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SRC))
from engine import decode_theta, make_state, make_world, save_payload, load_payload


def test_random_subspace_basis_is_orthonormal():
    _, p, *_ = make_world(31101, "aligned")
    np.testing.assert_allclose(p.T @ p, np.eye(4), atol=2e-6)


def test_mirror_codes_recover_aligned_tasks_and_coefficient_control():
    theta0, p, _, _, targets, xtr, ytr, *_ = make_world(31101, "aligned")
    mirror = make_state("mirror", theta0, p, xtr, ytr)
    coeff = make_state("coeff2", theta0, p, xtr, ytr)
    mirror_w = np.stack([decode_theta("mirror", mirror, t) for t in range(64)])
    coeff_w = np.stack([decode_theta("coeff2", coeff, t) for t in range(64)])
    assert np.mean((mirror_w - targets) ** 2) < 1e-4
    assert np.mean((coeff_w - targets) ** 2) < 1e-5


def test_unrelated_codes_need_more_than_one_angle():
    theta0, p, _, _, targets, xtr, ytr, *_ = make_world(31102, "unrelated")
    mirror = make_state("mirror", theta0, p, xtr, ytr)
    said = make_state("said4", theta0, p, xtr, ytr)
    mirror_w = np.stack([decode_theta("mirror", mirror, t) for t in range(64)])
    said_w = np.stack([decode_theta("said4", said, t) for t in range(64)])
    assert np.mean((mirror_w - targets) ** 2) > 0.005
    assert np.mean((said_w - targets) ** 2) < 1e-5


def test_serialized_payload_roundtrips_byte_exactly(tmp_path):
    theta0, p, _, _, _, xtr, ytr, *_ = make_world(31103, "aligned")
    state = make_state("mirror", theta0, p, xtr, ytr)
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    assert save_payload(a, state) == save_payload(b, state)
    assert a.read_bytes() == b.read_bytes()
    got = load_payload(a)
    for name in state:
        np.testing.assert_array_equal(got[name], state[name])
