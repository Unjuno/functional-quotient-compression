import io
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from model import Config, MoEViews, Teacher, rotate


def test_givens_view_is_norm_preserving_and_invertible():
    x = torch.randn(11, 16)
    angles = torch.tensor([0.2, -0.7, 0.4, 1.1])
    y = rotate(rotate(x, angles), angles, inverse=True)
    assert torch.allclose(x, y, atol=1e-6)
    assert torch.allclose(torch.linalg.vector_norm(x), torch.linalg.vector_norm(rotate(x, angles)), atol=1e-6)


def test_expert_pool_is_physically_tied_only_in_tied_methods():
    cfg = Config(residual_rank=1)
    assert MoEViews(cfg, "untied").parameter_count() > MoEViews(cfg, "tied").parameter_count()
    assert len(MoEViews(cfg, "tied").banks) == 1
    assert len(MoEViews(cfg, "untied").banks) == 2


def test_methods_emit_finite_outputs_and_have_serializable_payloads():
    cfg = Config(residual_rank=1)
    x = torch.randn(19, cfg.input_dim)
    layer = torch.arange(19) % cfg.layers
    for name in ("untied", "tied", "gate", "lowrank", "mirror"):
        model = MoEViews(cfg, name)
        y = model(x, layer)
        assert y.shape == (19, cfg.output_dim)
        assert torch.isfinite(y).all()
        assert model.inference_payload_bytes() > 0


def test_teacher_has_deterministic_targets_per_world():
    cfg = Config()
    x = torch.randn(64, cfg.input_dim)
    layer = torch.arange(64) % cfg.layers
    a, b = Teacher(cfg, 99), Teacher(cfg, 99)
    assert torch.equal(a(x, layer), b(x, layer))
