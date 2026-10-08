import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from model import Config, METHODS, Teacher, Top1MoE, rotate, route_labels


def test_givens_inverse_and_quadrant_routes():
    x = torch.randn(64, 16)
    a = torch.randn(64, 8) * 0.2
    torch.testing.assert_close(rotate(rotate(x, a), a, inverse=True), x, atol=2e-6, rtol=2e-6)
    labels = route_labels(x)
    assert set(labels.tolist()) == {0, 1, 2, 3}


def test_all_methods_roundtrip_serialized_outputs():
    cfg = Config()
    x = torch.randn(35, cfg.input_dim)
    for method in METHODS:
        torch.manual_seed(17)
        model = Top1MoE(cfg, method).eval()
        payload = model.serialize()
        loaded = torch.load(__import__("io").BytesIO(payload), map_location="cpu", weights_only=False)
        replica = Top1MoE(cfg, method).eval()
        replica.load_state_dict(loaded["state_dict"])
        with torch.no_grad():
            y, logits, role = model(x)
            yr, logits_r, role_r = replica(x)
        assert torch.equal(y, yr)
        assert torch.equal(logits, logits_r)
        assert torch.equal(role, role_r)
        assert len(payload) == model.inference_payload_bytes()


def test_teacher_modes_are_deterministic_and_distinct():
    cfg = Config()
    x = torch.randn(20, cfg.input_dim)
    y = route_labels(x)
    for mode in ("aligned", "independent"):
        a = Teacher(cfg, mode, 33)
        b = Teacher(cfg, mode, 33)
        torch.testing.assert_close(a.forward(x, y), b.forward(x, y), atol=0, rtol=0)
