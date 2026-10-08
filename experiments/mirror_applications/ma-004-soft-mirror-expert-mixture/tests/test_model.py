import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from model import Config, METHODS, Teacher, Top2MoE, rotate  # noqa: E402


def test_givens_inverse_round_trip():
    x = torch.randn(32, 16)
    angles = torch.randn(32, 4)
    assert torch.allclose(rotate(rotate(x, angles), angles, inverse=True), x, atol=2e-7)


def test_dense_soft_shapes_and_teacher_targets():
    cfg = Config()
    x = torch.randn(19, cfg.input_dim)
    teacher = Teacher(cfg, "aligned", 7)
    target, probs, indices = teacher.forward(x)
    assert target.shape == (19, cfg.output_dim)
    assert probs.shape == (19, cfg.experts)
    assert indices.shape == (19, 4)
    assert torch.allclose(probs.sum(-1), torch.ones(19))


def test_every_method_outputs_and_round_trips():
    cfg = Config()
    x = torch.randn(13, cfg.input_dim)
    for method in METHODS:
        model = Top2MoE(cfg, method)
        y, logits, indices, weights = model(x)
        assert y.shape == (13, cfg.output_dim)
        assert logits.shape == (13, cfg.experts)
        assert indices.shape == weights.shape == (13, 4)
        payload = model.serialize()
        loaded = torch.load(__import__("io").BytesIO(payload), weights_only=False)
        clone = Top2MoE(cfg, method)
        clone.load_state_dict(loaded["state_dict"])
        assert torch.equal(clone(x)[0], y)
