import importlib.util
from pathlib import Path
import numpy as np

MODULE = Path(__file__).parents[1] / "source" / "run_experiment.py"
spec = importlib.util.spec_from_file_location("ma550", MODULE)
ma550 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ma550)


def test_relative_error_zero_and_exact():
    x = np.array([1.0, 2.0], dtype=np.float32)
    assert ma550.rel_rmse(x, x) == 0.0
    assert ma550.rel_rmse(np.zeros_like(x), x) == 1.0


def test_world_has_pinned_heterogeneous_families():
    w = np.random.default_rng(7).normal(size=(40, ma550.WIDTH)).astype(np.float32)
    kinds, records = ma550.world(55001, w)
    assert kinds.count("sparse") == 8
    assert kinds.count("activation") == 8
    assert len(records) == 16
    assert all(q.shape == (40,) and s.shape == (8, 40) and h.shape == (8, 40) for q, s, h in records)


def test_safetensors_header_reader(tmp_path):
    import json, struct
    arr = np.arange(6, dtype="<f4").reshape(2, 3)
    header = json.dumps({"W": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    p = tmp_path / "x.safetensors"
    p.write_bytes(struct.pack("<Q", len(header)) + header + arr.tobytes())
    np.testing.assert_array_equal(ma550.read_safetensor(p, "W"), arr)


def test_safetensors_f16_reader(tmp_path):
    import json, struct
    arr = np.arange(6, dtype="<f2").reshape(2, 3)
    header = json.dumps({"W": {"dtype": "F16", "shape": [2, 3], "data_offsets": [0, 12]}}).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    p = tmp_path / "x.safetensors"
    p.write_bytes(struct.pack("<Q", len(header)) + header + arr.tobytes())
    np.testing.assert_array_equal(ma550.read_safetensor(p, "W"), arr.astype(np.float32))
