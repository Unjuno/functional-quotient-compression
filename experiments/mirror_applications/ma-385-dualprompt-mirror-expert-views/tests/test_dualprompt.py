import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from run import DualPromptModel, evaluate, make_data, write_model_payload  # noqa: E402


def test_train_key_centroid_router_recovers_sequential_task_address():
    data = make_data(38501)
    model = DualPromptModel("hard_shared", 1)
    keys = torch.tensor(data["router_keys"])
    assert evaluate(model, data["test"], keys)["retrieval_accuracy"] >= 0.98


def test_mirror_views_share_general_prompt_and_rank_two_expert_basis():
    model = DualPromptModel("mirror_angle", 2)
    prompts = model.prompts()
    coeff = torch.stack([model.angles.cos(), model.angles.sin()], dim=1)
    expected = model.general[None, :] + coeff @ model.basis.T
    torch.testing.assert_close(prompts, expected)
    assert prompts.shape == (8, 16)


def test_payload_charges_general_prompt_expert_view_and_router(tmp_path):
    data = make_data(38502)
    model = DualPromptModel("mirror_angle", 3)
    arrays, size, digest = write_model_payload(model, data, tmp_path / "payload.npz")
    assert "general" in arrays and "basis" in arrays and "angles" in arrays and "router_keys" in arrays
    assert size == (tmp_path / "payload.npz").stat().st_size and size > 0
    assert len(digest) == 64
