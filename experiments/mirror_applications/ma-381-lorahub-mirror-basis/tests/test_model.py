import sys
from pathlib import Path

import torch

SOURCE = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))

from model import CANDIDATES, RANK, LoRABank, rotation_matrices  # noqa: E402
from run import gauge_audit  # noqa: E402


def test_so2_code_is_orthogonal_and_differentiable():
    angles = torch.randn(CANDIDATES, 1, requires_grad=True)
    q = rotation_matrices(angles)
    eye = torch.eye(RANK).expand(CANDIDATES, RANK, RANK)
    assert torch.allclose(q.transpose(-1, -2) @ q, eye, atol=1e-6)
    (q[:, 0, 0].sum()).backward()
    assert angles.grad is not None


def test_all_source_bank_variants_and_lorahub_composition():
    x = torch.randn(13, 16)
    for method in LoRABank.METHODS:
        bank = LoRABank(method, 23)
        candidates = bank(x)
        assert candidates.shape == (13, CANDIDATES, 16)
        coeff = torch.randn(4, CANDIDATES)
        composed = torch.einsum("tn,bnd->tbd", coeff, candidates)
        assert composed.shape == (4, 13, 16)
        composed.square().mean().backward()
        assert all(p.grad is not None for p in bank.parameters())


def test_nonorthogonal_lora_factor_gauge_preserves_full_function():
    bank = LoRABank("independent", seed=31)
    gauge = gauge_audit(bank, 19)
    assert gauge["max_delta_matrix"] < 1e-6
    assert gauge["max_delta_output"] < 1e-6
    assert gauge["max_delta_composed"] < 1e-6
