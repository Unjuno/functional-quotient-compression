import importlib.util
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "source" / "run.py"
spec = importlib.util.spec_from_file_location("ma304_run", SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_teacher_has_fixed_sparse_active_support():
    w = m.make_world(30401)
    assert np.count_nonzero(w["b1"]) == m.K
    assert np.count_nonzero(w["b2"]) == m.K
    assert all(np.count_nonzero(x) == m.K for x in w["weights"])


def test_direct_and_mirror_recover_aligned_function():
    w = m.make_world(30401)
    for method in ("direct_pair_private", "mirror_phase_private"):
        arrays, _, _ = m.encode(method, w)
        assert len(arrays["private_ids"]) == m.N_PRIVATE
        for i in (0, 127):
            pred = m.decode(method, arrays, i, w)
            assert m.nrmse(pred, w["weights"][i]) < 1e-3


def test_unrelated_task_uses_private_fallback():
    w = m.make_world(30401)
    arrays, _, _ = m.encode("mirror_phase_private", w)
    assert np.array_equal(arrays["private_ids"], np.arange(m.N_ALIGNED, m.N_TASKS))


def test_payload_roundtrip_hash_is_deterministic(tmp_path):
    w = m.make_world(30401)
    arrays, _, _ = m.encode("mirror_phase_private", w)
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    m.pack(arrays, a)
    m.pack(arrays, b)
    assert a.read_bytes() == b.read_bytes()
    loaded = m.unpack(a)
    assert np.array_equal(loaded["private_ids"], arrays["private_ids"])
