from pathlib import Path
import sys

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SRC))
from engine import D, K, T, decode_theta, make_state, make_world, save_payload, load_payload


def test_shared_basis_is_orthonormal_and_shared_across_many_tasks():
    _, basis, _, _, _, *_ = make_world(31201, "aligned")
    assert basis.shape == (D, K)
    np.testing.assert_allclose(basis.T @ basis, np.eye(K), atol=2e-6)
    assert T == 256


def test_mirror_recovers_aligned_many_task_bank():
    theta0, basis, _, _, targets, xtr, ytr, *_ = make_world(31201, "aligned")
    state = make_state("mirror", theta0, basis, xtr, ytr)
    weights = np.stack([decode_theta("mirror", state, t) for t in range(T)])
    assert np.mean((weights - targets) ** 2) < 1e-4


def test_unrelated_tasks_need_full_intrinsic_codes():
    theta0, basis, _, _, targets, xtr, ytr, *_ = make_world(31202, "unrelated")
    mirror = make_state("mirror", theta0, basis, xtr, ytr)
    said = make_state("said8", theta0, basis, xtr, ytr)
    wm = np.stack([decode_theta("mirror", mirror, t) for t in range(T)])
    ws = np.stack([decode_theta("said8", said, t) for t in range(T)])
    assert np.mean((wm - targets) ** 2) > 0.005
    assert np.mean((ws - targets) ** 2) < 1e-5


def test_payload_is_byte_stable_and_roundtrips(tmp_path):
    theta0, basis, _, _, _, xtr, ytr, *_ = make_world(31203, "aligned")
    state = make_state("mirror", theta0, basis, xtr, ytr)
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    assert save_payload(a, state) == save_payload(b, state)
    assert a.read_bytes() == b.read_bytes()
    loaded = load_payload(a)
    for key in state:
        np.testing.assert_array_equal(loaded[key], state[key])
