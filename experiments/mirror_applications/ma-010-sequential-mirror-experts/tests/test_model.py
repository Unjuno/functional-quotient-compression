import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from engine import apply_sequence, make_teacher
from model import SequentialExperts


def test_aligned_teacher_is_in_mirror_orbit():
    mats = make_teacher("aligned", 20261007)
    model = SequentialExperts("mirror", 19)
    with torch.no_grad():
        # Copy exact teacher base and view coordinates to establish expressive coverage.
        g = torch.Generator().manual_seed(20261007)
        base = torch.randn(8, 8, generator=g) * 0.10 + torch.eye(8) * 0.18
        ain = torch.rand(4, generator=g) * 1.4 - 0.7
        aout = torch.rand(4, generator=g) * 1.4 - 0.7
        model.weight.copy_(base)
        model.input_angle.copy_(ain)
        model.output_angle.copy_(aout)
    assert torch.allclose(model.matrices(), mats, atol=1e-7)


def test_sequence_order_is_functional():
    mats = make_teacher("independent", 55)
    x = torch.randn(16, 8, generator=torch.Generator().manual_seed(12))
    roles = torch.tensor([[0, 1]] * len(x))
    ordered = apply_sequence(mats, x, roles)
    reversed_output = apply_sequence(mats, x, roles.flip(1))
    assert torch.mean((ordered - reversed_output) ** 2).item() > 1e-7


def test_all_models_train_and_report_four_role_matrices():
    x = torch.randn(5, 8)
    roles = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 0]])
    for method in ("independent", "tied", "rank2", "scalar", "mirror"):
        model = SequentialExperts(method, 100)
        loss = model(x, roles).square().mean()
        loss.backward()
        assert model.matrices().shape == (4, 8, 8)
        assert all(p.grad is not None for p in model.parameters())
