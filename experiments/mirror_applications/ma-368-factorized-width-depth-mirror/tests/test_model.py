import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from model import DEPTHS, HELD_OUT, WIDTHS, ElasticSupernet  # noqa: E402
from run import TRAIN_CONFIGS, sampled_configs  # noqa: E402


def test_factorized_code_has_one_value_per_width_and_depth():
    model = ElasticSupernet("factor_mirror", 10)
    assert model.width_codes.numel() == len(WIDTHS)
    assert model.depth_codes.numel() == len(DEPTHS)
    assert model.width_codes.numel() + model.depth_codes.numel() == 8


def test_shared_logits_cover_grid_and_heldout_configs_are_not_sampled():
    model = ElasticSupernet("factor_mirror", 11)
    x = torch.randn(3, 32)
    for width in WIDTHS:
        for depth in DEPTHS:
            assert model.forward_config(width, depth, x).shape == (3, 4)
    assert all(config not in sampled_configs(123, step) for config in HELD_OUT for step in range(50))
    assert set(TRAIN_CONFIGS).isdisjoint(set(HELD_OUT))


def test_mirror_width_depth_code_changes_function():
    model = ElasticSupernet("factor_mirror", 12)
    x = torch.randn(4, 32)
    before = model.forward_config(16, 3, x).detach().clone()
    with torch.no_grad():
        model.width_codes[1] = 0.25
    after = model.forward_config(16, 3, x)
    assert not torch.allclose(before, after)
