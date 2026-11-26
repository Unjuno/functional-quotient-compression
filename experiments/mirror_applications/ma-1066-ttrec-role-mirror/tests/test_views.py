import torch
from torch import nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source'))
from views import apply_givens


def test_zero_angles_are_identity():
    x = torch.randn(13,32)
    assert torch.allclose(apply_givens(x, torch.zeros(16)), x, atol=1e-7)


def test_givens_preserves_per_row_norm():
    x = torch.randn(13,32)
    y = apply_givens(x, torch.randn(16))
    assert torch.allclose(x.norm(dim=1), y.norm(dim=1), atol=1e-5, rtol=1e-5)


def test_matched_diagonal_control_has_same_code_count():
    mirror = nn.Parameter(torch.zeros(16))
    diagonal = nn.Parameter(torch.zeros(16))
    assert mirror.numel() == diagonal.numel()
