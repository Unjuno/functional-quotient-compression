import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from run import HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM, maps, output_map, pair_rotate, teacher_world
from verify import verify_world


def test_givens_preserves_norm():
    rng = np.random.default_rng(22)
    h = rng.normal(size=(32, HIDDEN_DIM))
    rotated = pair_rotate(h, 0.73)
    assert np.max(np.abs(np.sum(h * h, axis=1) - np.sum(rotated * rotated, axis=1))) < 1e-12


def test_width_operator_matches_pair_rotation():
    rng = np.random.default_rng(23)
    U = rng.normal(size=(INPUT_DIM, HIDDEN_DIM))
    V = rng.normal(size=(HIDDEN_DIM, OUTPUT_DIM))
    x = rng.normal(size=(7, INPUT_DIM))
    k, theta = 20, -0.31
    h = pair_rotate(x @ U[:, :k], theta)
    expected = h @ V[:k]
    actual = x @ output_map(U, V, k, theta)
    assert np.max(np.abs(expected - actual)) < 1e-10


def test_teacher_world_has_locked_width_split():
    world = teacher_world(56001)
    assert set(world["data"]) == {4, 8, 12, 16, 20, 24, 28, 32}
    assert world["data"][4][0].shape == (512, 16)
    assert world["data"][4][2].shape == (512, 16)


def test_saved_development_payloads_replay_exactly():
    root = ROOT / "runs" / "dev"
    worlds = sorted(root.glob("*/RESULTS_CORE.csv"))
    assert worlds, "run frozen development worlds before artifact replay"
    assert sum(verify_world(path.parent) for path in worlds) == 32 * len(worlds)
