import importlib.util
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "source" / "run_ma1064.py"
SPEC = importlib.util.spec_from_file_location("ma1064_runner", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_fixed_hash_is_deterministic_and_in_range():
    for item in (1, 7, 943, 1682):
        for seed in RUNNER.HASH_SEEDS:
            value = RUNNER.hash_bucket(item, seed)
            assert value == RUNNER.hash_bucket(item, seed)
            assert 0 <= value < RUNNER.BUCKETS


def test_zero_mirror_code_is_identity_for_same_dhe_weights():
    rare = np.array([2, 8], dtype=np.int64)
    dhe = RUNNER.build("dhe", rare, seed=31).eval()
    mirror = RUNNER.build("dhe_mirror", rare, seed=31).eval()
    items = torch.tensor([0, 2, 8, 10], dtype=torch.long)
    with torch.no_grad():
        base = dhe.item_vector(items)
        viewed = mirror.item_vector(items)
    assert torch.allclose(base, viewed, atol=1e-7, rtol=1e-7)


def test_mirror_rotation_preserves_norm_without_private_residual():
    rare = np.array([2, 8], dtype=np.int64)
    model = RUNNER.build("dhe_mirror", rare, seed=47).eval()
    with torch.no_grad():
        model.mirror_angles.fill_(0.7)
        item_ids = torch.tensor([2, 8], dtype=torch.long)
        base = model._base_item_vector(item_ids)
        viewed = model.item_vector(item_ids)
    assert torch.allclose(base.norm(dim=1), viewed.norm(dim=1), atol=1e-6, rtol=1e-6)


def test_rare_id_mapping_only_applies_to_declared_ids():
    rare = np.array([2, 8], dtype=np.int64)
    model = RUNNER.build("dhe_mirror", rare, seed=59).eval()
    with torch.no_grad():
        model.mirror_angles.fill_(0.5)
        item_ids = torch.tensor([0, 3, 9], dtype=torch.long)
        base = model._base_item_vector(item_ids)
        viewed = model.item_vector(item_ids)
    assert torch.equal(base, viewed)


def test_dhe_inference_payload_does_not_charge_unused_rare_id_list():
    rare = np.array([2, 8], dtype=np.int64)
    dhe = RUNNER.build("dhe", rare, seed=31)
    mirror = RUNNER.build("dhe_mirror", rare, seed=31).eval()
    assert "rare_item_ids_one_based" not in dhe.inference_state()
    assert "rare_item_ids_one_based" in mirror.inference_state()


def test_payload_metadata_matches_actual_method():
    data = {"archive_sha256": "frozen"}
    dhe = RUNNER.model_meta("dhe", 31, np.array([2]), 2, data)
    full = RUNNER.model_meta("full_table", 31, np.array([2]), 2, data)
    mirror = RUNNER.model_meta("dhe_mirror", 31, np.array([2]), 2, data)
    assert dhe["mirror"] == "none" and "hash" in dhe
    assert "hash" not in full and "decoder" not in full
    assert "Givens angle" in mirror["mirror"]
