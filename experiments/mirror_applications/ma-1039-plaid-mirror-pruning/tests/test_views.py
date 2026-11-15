import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from plaid_views import rotate_rows


def test_givens_preserves_norm_and_inverse():
    torch.manual_seed(1039)
    x = torch.randn(11, 32, 128)
    y = rotate_rows(x, 0.17)
    z = rotate_rows(y, -0.17)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=2e-6, rtol=2e-6)
    assert torch.allclose(x, z, atol=3e-7, rtol=3e-7)


def test_centroid_bank_view_equals_native_query_pushthrough():
    torch.manual_seed(1040)
    c = torch.nn.functional.normalize(torch.randn(512, 128), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(32, 128), dim=-1)
    angle = 0.2
    bank_scores = rotate_rows(c, angle) @ q.T
    query_scores = c @ rotate_rows(q, -angle).T
    assert torch.allclose(bank_scores, query_scores, atol=2e-6, rtol=2e-6)
    assert torch.equal(bank_scores.topk(2, dim=0).indices, query_scores.topk(2, dim=0).indices)


def test_zero_angle_is_exact_identity():
    x = torch.randn(2, 128)
    assert torch.equal(x, rotate_rows(x, 0.0))


class _Codec:
    def __init__(self, centroids):
        self.centroids = centroids


class _Ranker:
    def __init__(self, centroids):
        self.codec = _Codec(centroids)

    def get_cells(self, q, ncells):
        scores = self.codec.centroids @ q.T
        return scores.topk(ncells, dim=0, sorted=False).indices.T.flatten().unique(), scores

    def generate_candidate_pids(self, q, ncells):
        cells, scores = self.get_cells(q, ncells)
        return cells, scores


def test_adapter_records_matching_cells_and_candidates():
    from plaid_views import configure_state, install_view_adapter
    torch.manual_seed(1041)
    ranker = _Ranker(torch.nn.functional.normalize(torch.randn(64, 128), dim=-1))
    state = install_view_adapter(ranker)
    q = torch.nn.functional.normalize(torch.randn(32, 128), dim=-1)
    configure_state(state, "mirror_centroid_bank", 0.15)
    mirror_pids, mirror_scores = ranker.generate_candidate_pids(q, 2)
    mirror_cells = state["cells"].clone()
    configure_state(state, "native_query_routing", 0.15)
    native_pids, native_scores = ranker.generate_candidate_pids(q, 2)
    assert torch.equal(mirror_cells, state["cells"])
    assert torch.equal(mirror_pids, native_pids)
    assert torch.allclose(mirror_scores, native_scores, atol=2e-6, rtol=2e-6)
    assert torch.equal(ranker.codec.centroids, rotate_rows(ranker.codec.centroids, 0.0))


def test_seeded_training_sample_is_stable_and_bounded():
    import random
    def sample(n, seed=1039):
        return random.Random(seed).sample(range(n), min(4096, n))
    first = sample(18049)
    assert first == sample(18049)
    assert len(first) == 4096 and len(set(first)) == 4096
    assert sample(40) == sample(40) and len(sample(40)) == 40
