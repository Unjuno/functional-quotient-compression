import sys
from pathlib import Path

import torch

SOURCE = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))

from model import PLANES, RANK, SOURCES, SourceBank, mac_proxy, rotation_matrices  # noqa: E402


def test_givens_composition_is_orthogonal_and_differentiable():
    angles = torch.randn(SOURCES, len(PLANES), requires_grad=True)
    q = rotation_matrices(angles)
    eye = torch.eye(RANK).expand(SOURCES, RANK, RANK)
    assert torch.allclose(q.transpose(-1, -2) @ q, eye, atol=1e-6)
    q.square().sum().backward()
    assert angles.grad is not None


def test_all_adapter_banks_return_expected_shape_and_gradients():
    x = torch.randn(11, 16)
    for method in SourceBank.METHODS:
        bank = SourceBank(method, seed=7)
        y = bank(x)
        assert y.shape == (11, SOURCES, 16)
        y.square().mean().backward()
        assert all(p.grad is not None for p in bank.parameters())


def test_tied_bank_has_identical_source_views():
    bank = SourceBank("tied", seed=9)
    y = bank(torch.randn(5, 16))
    for i in range(1, SOURCES):
        assert torch.equal(y[:, 0], y[:, i])


def test_source_mac_proxy_excludes_fusion_router():
    assert mac_proxy("independent") == 8 * 2 * 16 * 4
    assert mac_proxy("mirror") == 2 * 16 * 4 + 8 * 4 * 6 + 8 * 16 * 4
