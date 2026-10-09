import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from model import HashEmbeddingBank, hash_bucket, hash_indices  # noqa: E402
from run import make_world, score, save_payload, load_payload  # noqa: E402


def test_hash_is_stable_and_has_expected_shape():
    assert hash_bucket(17, 991) == hash_bucket(17, 991)
    assert hash_indices(1, 2).shape == (256, 2)
    assert int(hash_indices(1, 2).min()) >= 0
    assert int(hash_indices(1, 2).max()) < 32


def test_mirror_coordinate_reconstructs_unit_circle_weights():
    world = make_world(38901)
    bank = HashEmbeddingBank("mirror", 38902, world["hash_seed0"], world["hash_seed1"], world["decoder"])
    alpha = bank.importance_values()
    torch.testing.assert_close(torch.sum(alpha * alpha, dim=1), torch.ones(256), atol=2e-7, rtol=0)


def test_serialized_payload_reloads_to_identical_scores(tmp_path):
    world = make_world(38901)
    bank = HashEmbeddingBank("mirror", 38903, world["hash_seed0"], world["hash_seed1"], world["decoder"])
    path = tmp_path / "mirror.npz"
    save_payload(path, bank)
    loaded = load_payload(path)
    assert loaded.method == "mirror"
    assert score(loaded, world) == score(bank, world)


def test_methods_are_distinct_control_shapes():
    world = make_world(38902)
    shapes = {}
    for method in HashEmbeddingBank.METHODS:
        bank = HashEmbeddingBank(method, 7, world["hash_seed0"], world["hash_seed1"], world["decoder"])
        shapes[method] = sum(p.numel() for p in bank.parameters())
    assert shapes["full"] == 256 * 16
    assert shapes["hash"] == 2 * 32 * 16 + 256 * 2
    assert shapes["mirror"] == 2 * 32 * 16 + 256
    assert shapes["mirror"] < shapes["hash"]
