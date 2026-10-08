import importlib.util
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "source" / "run.py"
spec = importlib.util.spec_from_file_location("ma309_run", SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_rule_labels_cover_four_distinct_tasks():
    x = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [2.0, 2.0]])
    labels = [m.rule_labels(x, i).tolist() for i in range(4)]
    assert labels[0] == [1, 0, 1]
    assert labels[1] == [0, 1, 1]
    assert labels[2] == [0, 0, 1]
    assert labels[3] == [0, 0, 1]


def test_mirror_view_changes_member_function():
    torch.manual_seed(9)
    model = m.MimoModel("mirror_givens")
    x = torch.randn(4, 16, 2)
    before = model(x).detach()
    with torch.no_grad():
        model.angles[0] = 1.1
    after = model(x).detach()
    assert not torch.equal(before[0], after[0])
    assert torch.equal(before[1:], after[1:])


def test_independent_upper_has_member_local_trunks_and_heads():
    model = m.MimoModel("independent_mlps")
    names = list(model.state_dict())
    assert any(n.startswith("trunks.1") for n in names)
    assert any(n.startswith("heads.1") for n in names)
    assert not any(n.startswith("head.") for n in names)


def test_serialized_payload_is_deterministic(tmp_path):
    torch.manual_seed(3)
    model = m.MimoModel("mirror_givens")
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    m.payload(model, a)
    m.payload(model, b)
    assert a.read_bytes() == b.read_bytes()


def test_reload_inference_model_uses_payload_weights(tmp_path):
    torch.manual_seed(31)
    model = m.MimoModel("mirror_givens")
    _, _, arrays = m.payload(model, tmp_path / "payload.npz")
    restored = m.reload_inference_model("mirror_givens", arrays)
    for key, value in restored.state_dict().items():
        expected = torch.from_numpy(arrays[key].copy()).to(dtype=value.dtype)
        assert torch.equal(value, expected)
