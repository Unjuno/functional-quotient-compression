import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from model import SharedSlimmableMLP, WIDTHS  # noqa: E402


def test_all_widths_have_expected_logits():
    model = SharedSlimmableMLP("usnet", 1)
    x = torch.randn(5, 32)
    for width in WIDTHS:
        assert model.forward_width(width, x).shape == (5, 4)


def test_mirror_coordinate_changes_function():
    model = SharedSlimmableMLP("mirror_givens", 2)
    x = torch.randn(5, 32)
    before = model.forward_width(16, x).detach().clone()
    with torch.no_grad():
        model.codes[1] = 0.3
    after = model.forward_width(16, x)
    assert not torch.allclose(before, after)


def test_width_slice_excludes_inactive_channels():
    model = SharedSlimmableMLP("usnet", 3)
    x = torch.randn(2, 32)
    before = model.forward_width(8, x).detach().clone()
    with torch.no_grad():
        model.w1[:, 8:] += 1000
        model.w2[8:, :] -= 1000
    after = model.forward_width(8, x)
    assert torch.equal(before, after)
