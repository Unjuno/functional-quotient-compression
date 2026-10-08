import importlib.util
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "source" / "run.py"
spec = importlib.util.spec_from_file_location("ma318_run", SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_stream_uses_three_planted_function_groups():
    w = m.make_world(31801)
    assert w["weights"].shape == (12, 64)
    assert w["groups"] == ["initial_orbit"] * 4 + ["second_orbit"] * 4 + ["unrelated"] * 4


def test_phase_coordinate_recovers_first_orbit():
    w = m.make_world(31801)
    x, y = w["sets"][0][0]
    a, _ = m.fit_phase(x, y, w["initial_basis"])
    pred = np.cos(a) * w["initial_basis"][0] + np.sin(a) * w["initial_basis"][1]
    assert m.nrmse(w["sets"][0][2][0] @ pred, w["sets"][0][2][1]) < 1e-4


def test_new_skill_growth_preserves_previous_tasks(tmp_path):
    w = m.make_world(31801)
    rows, _ = m.run_method("mirror_coordinate_first", w, "development", tmp_path)
    assert rows[3]["basis_growth_count"] == 0
    assert rows[7]["basis_growth_count"] == 2
    assert rows[-1]["basis_growth_count"] == 6
    assert max(x["max_prior_test_n_mse"] for x in rows) < 1e-4


def test_payload_bytes_and_hash_roundtrip(tmp_path):
    arrays = {"a": np.arange(5, dtype=np.float16), "b": np.asarray([1], np.uint8)}
    p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
    n1, h1 = m.pack(arrays, p1)
    n2, h2 = m.pack(arrays, p2)
    assert n1 == n2 and h1 == h2 and p1.read_bytes() == p2.read_bytes()
    assert np.array_equal(m.unpack(p1)["a"], arrays["a"])
