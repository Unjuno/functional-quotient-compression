import sys
from pathlib import Path

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))
import run


def test_givens_map_is_task_function_and_angle_changes_it():
    template = run.TEMPLATE
    zero = run.orbit(torch.zeros(2))
    turned = run.orbit(torch.tensor([0.3, -0.4]))
    assert torch.equal(zero, template)
    assert torch.linalg.vector_norm(turned - zero).item() > 0.1


def test_mirror_and_native_givens_have_same_functional_map():
    base = torch.tensor([0.2, -0.1, 0.4, 0.1, 0.0, 0.3])
    code = torch.tensor([-0.2, 0.6])
    mirror = run.prediction_weight("mirror", base, code)
    native = run.prediction_weight("native_givens", base, code)
    assert torch.equal(mirror, native)


def test_npz_payload_roundtrip_preserves_paid_arrays():
    original = {"shared": np.arange(6, dtype=np.float32), "codes": np.ones((4, 2), dtype=np.float32)}
    payload = run.serialize_payload(original, {"format": "test"})
    loaded = np.load(__import__("io").BytesIO(payload))
    assert np.array_equal(loaded["shared"], original["shared"])
    assert np.array_equal(loaded["codes"], original["codes"])
    assert len(payload) > sum(v.nbytes for v in original.values())
