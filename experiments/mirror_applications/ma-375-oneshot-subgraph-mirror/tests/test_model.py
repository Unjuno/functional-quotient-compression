import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from model import METHODS, PATHS, SharedDAG  # noqa: E402
from run import spearman  # noqa: E402


def test_all_paths_are_functionally_supported():
    model = SharedDAG("supernet", 1)
    x = torch.randn(4, 16)
    assert len(PATHS) == 8
    assert all(model.forward_path(path, x).shape == (4, 4) for path in PATHS)


def test_shared_nodes_are_one_physical_bank():
    model = SharedDAG("supernet", 2)
    assert len(model.nodes) == 3
    assert all(model.nodes[i] is not model.nodes[j] for i in range(3) for j in range(i + 1, 3))


def test_mirror_path_coordinate_changes_function():
    model = SharedDAG("mirror_path", 3)
    x = torch.randn(4, 16)
    before = model.forward_path(5, x).detach().clone()
    with torch.no_grad():
        model.codes[5] = 0.3
    assert not torch.allclose(before, model.forward_path(5, x))


def test_spearman_detects_ranking_and_ties():
    assert spearman([1, 2, 3], [1, 2, 3]) == 1.0
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0
