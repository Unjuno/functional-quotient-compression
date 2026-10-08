import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from run import DIM, HashModel, indices_from_params, make_data, model_arrays, deterministic_pack  # noqa: E402


def test_hash_pair_collisions_are_explicitly_counted():
    data = make_data(38901)
    assert 0.0 < data["collision_rate"] < 1.0
    assert data["collision_groups"] > 0
    assert data["hash_indices"].shape == (128, 2)


def test_mirror_is_one_angle_over_the_same_two_hash_components():
    data = make_data(38902)
    model = HashModel("mirror_angle", 1)
    indices = torch.tensor(data["hash_indices"], dtype=torch.long)
    tokens = torch.arange(128)
    got = model.embed(tokens, indices)
    e0 = model.table0[indices[:, 0]]
    e1 = model.table1[indices[:, 1]]
    want = torch.cos(model.angles)[:, None] * e0 + torch.sin(model.angles)[:, None] * e1
    torch.testing.assert_close(got, want)
    assert got.shape == (128, DIM)


def test_payload_charges_importance_code_and_hash_parameters(tmp_path):
    data = make_data(38901)
    model = HashModel("mirror_angle", 7)
    arrays = model_arrays(model, data, 38901)
    size, digest = deterministic_pack(arrays, tmp_path / "payload.npz")
    assert "angles" in arrays and "table0" in arrays and "table1" in arrays
    assert "hash_params" in arrays and len(arrays["hash_params"]) == 2
    assert size == (tmp_path / "payload.npz").stat().st_size and size > 0
    assert len(digest) == 64
    assert indices_from_params(arrays["hash_params"]).shape == (128, 2)
