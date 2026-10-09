import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "source"))
from model import Config, METHODS, ExpertChoiceMoE, assignments, rotate  # noqa: E402
from engine import make_balanced  # noqa: E402


def test_expert_choice_enforces_per_expert_capacity_and_token_choice_is_one_hot():
    cfg = Config()
    logits = torch.randn(64, cfg.experts)
    mask, weights = assignments(logits, "expert_choice", cfg.capacity_per_expert)
    assert mask.sum(dim=0).tolist() == [16, 16, 16, 16]
    assert torch.allclose(weights.sum(-1)[mask.any(-1)], torch.ones(int(mask.any(-1).sum())))
    assert torch.all(weights[~mask] == 0)
    token_mask, token_weights = assignments(logits, "token", cfg.capacity_per_expert)
    assert torch.all(token_mask.sum(-1) == 1)
    assert torch.allclose(token_weights.sum(-1), torch.ones(64))


def test_balanced_data_has_fixed_per_batch_role_counts():
    cfg = Config()
    x, labels = make_balanced(11, 64 * 3, cfg)
    assert x.shape == (192, cfg.input_dim)
    for batch in labels.view(3, 64):
        assert torch.bincount(batch, minlength=4).tolist() == [16, 16, 16, 16]


def test_all_methods_forward_train_and_serialize():
    cfg = Config()
    x, labels = make_balanced(12, 64, cfg)
    angles = torch.randn(4, 4)
    z = torch.randn(32, 16)
    assert torch.allclose(rotate(rotate(z, angles[0]), angles[0], inverse=True), z, atol=2e-7)
    token_mirror = ExpertChoiceMoE(cfg, "mirror_token")
    _, _, token_mask, _ = token_mirror(x)
    assert torch.all(token_mask.sum(-1) == 1)
    for method in METHODS:
        model = ExpertChoiceMoE(cfg, method)
        y, logits, mask, weights = model(x)
        assert y.shape == (64, cfg.output_dim)
        assert logits.shape == mask.shape == weights.shape == (64, cfg.experts)
        loss = y.square().mean() + torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        payload = model.serialize()
        loaded = torch.load(__import__("io").BytesIO(payload), weights_only=False)
        clone = ExpertChoiceMoE(cfg, method)
        clone.load_state_dict(loaded["state_dict"])
        assert torch.equal(clone(x)[0], y)
