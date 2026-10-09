import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from engine import DEPTH, WIDTH, DepthModel, make_world, serialized_bytes, serialized_payload  # noqa: E402


def test_mirror_model_builds_depth_specific_matrices_and_gradients():
    model = DepthModel("mirror")
    x = torch.randn(5, WIDTH)
    model(x).square().mean().backward()
    assert model.matrices().shape == (DEPTH, WIDTH, WIDTH)
    assert model.angle.grad is not None
    assert torch.isfinite(model.angle.grad).all()


def test_world_shapes_and_payload_are_deterministic():
    a = make_world(76, aligned=True)
    b = make_world(76, aligned=True)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert a[1].shape == (DEPTH, 256, WIDTH)
    assert serialized_bytes(DepthModel("tied")) == serialized_bytes(DepthModel("tied"))


def test_serialized_inference_state_roundtrips():
    model = DepthModel("mirror")
    payload = torch.load(__import__("io").BytesIO(serialized_payload(model)), weights_only=False)
    restored = DepthModel(payload["method"])
    restored.load_state_dict(payload["state_dict"])
    x = torch.randn(7, WIDTH)
    assert torch.equal(model(x), restored(x))
