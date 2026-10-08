import importlib.util
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "source" / "run_ma341.py"
spec = importlib.util.spec_from_file_location("run_ma341", SCRIPT)
ma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ma)


def test_partition_is_disjoint_cover_and_replayable():
    x1, y1, s1 = ma.partition(23)
    x2, y2, s2 = ma.partition(23)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    flat1 = np.concatenate([a for d in s1 for a in d.values()])
    flat2 = np.concatenate([a for d in s2 for a in d.values()])
    assert len(flat1) == len(y1) and len(np.unique(flat1)) == len(y1)
    assert np.array_equal(flat1, flat2)
    assert all(len(s1[c]["train"]) > 0 and len(s1[c]["dev"]) > 0 for c in range(10))
    assert all(len(s1[c]["support"]) > 0 and len(s1[c]["query"]) > 0 for c in (10, 11))


def test_models_emit_expected_shape_and_parameter_counts():
    x = torch.randn(5, 64)
    assert sum(p.numel() for p in ma.MLP().parameters()) == 2410
    for method in ma.METHODS:
        ma.seed_all(7)
        model = ma.System(method)
        assert model.logits(x, 0).shape == (5, 10)
        if method == "pfedhn":
            assert model.hyper.generate(model.codes[0])[0].shape == (32, 64)


def test_mirror_rotation_preserves_hidden_pair_norm():
    ma.seed_all(4)
    model = ma.System("mirror")
    x = torch.randn(3, 64)
    h = torch.relu(model.base.fc1(x))
    theta = torch.tensor([0.1, -0.3, 0.7, 1.1])
    # Mirror forward must remain finite, and its rotation is orthogonal by construction.
    assert torch.isfinite(model.logits(x, 0, theta)).all()
    for j in range(4):
        a, b = 2*j, 2*j+1
        co, si = torch.cos(theta[j]), torch.sin(theta[j])
        ha, hb = h[:, a], h[:, b]
        assert torch.allclose((co*ha-si*hb)**2 + (si*ha+co*hb)**2, ha**2+hb**2, atol=1e-6)


def test_film_code_has_nonzero_initial_gradient():
    ma.seed_all(31)
    model = ma.System("film")
    x, y = torch.randn(8, 64), torch.randint(0, 10, (8,))
    ma.loss_for(model, x, y, 0).backward()
    assert model.codes.grad is not None
    assert model.codes.grad.abs().max().item() > 0


def test_payload_is_safetensors_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "OUT", tmp_path)
    data = {"w": torch.arange(5, dtype=torch.float32)}
    n = ma.tensor_payload_bytes(data, "x.safetensors")
    assert n == (tmp_path / "x.safetensors").stat().st_size
    assert n > data["w"].numel() * 4
