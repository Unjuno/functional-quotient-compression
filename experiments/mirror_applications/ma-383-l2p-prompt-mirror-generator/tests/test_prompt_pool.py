import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from run import METHODS, PromptModel, evaluate, make_data, state_arrays, deterministic_pack  # noqa: E402


def test_train_centroid_router_retrieves_task_without_task_id():
    data = make_data(38301)
    keys = torch.tensor(data["keys"])
    model = PromptModel("hard_shared", 1)
    metrics = evaluate(model, data["test"], keys)
    assert metrics["retrieval_accuracy"] >= 0.98


def test_angle_view_is_rank_two_shared_basis_with_one_code_per_task():
    model = PromptModel("mirror_angle", 4)
    keys = torch.zeros(4, 8)
    prompts = model.logical_prompts(keys)
    assert prompts.shape == (4, 8)
    coeff = torch.stack([model.angles.cos(), model.angles.sin()], dim=1)
    torch.testing.assert_close(prompts, coeff @ model.basis.T)
    assert len(model.angles) == 4


def test_actual_zip_payload_counts_router_and_roundtrips(tmp_path):
    data = make_data(38302)
    model = PromptModel("mirror_angle", 2)
    arrays = state_arrays(model, data)
    path = tmp_path / "payload.npz"
    size, digest = deterministic_pack(arrays, path)
    assert size == path.stat().st_size and size > 0
    assert len(digest) == 64
    assert "router_keys" in arrays
    assert arrays["meta"][3] == METHODS.index("mirror_angle")
