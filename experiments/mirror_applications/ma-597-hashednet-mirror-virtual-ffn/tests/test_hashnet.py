import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from run_hashednet import collision_metrics, givens_rotate, hash_map


def test_hash_map_and_collision_audit_are_deterministic():
    a = hash_map(16, 73)
    b = hash_map(16, 73)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    m = collision_metrics(a[0], 16)
    assert m['unique_bucket_count'] <= 16
    assert m['pairwise_collision_count'] >= 0
    assert 0 <= m['collision_rate'] <= 1


def test_givens_view_preserves_input_norm():
    torch.manual_seed(11)
    x = torch.randn(12, 64)
    angles = torch.randn(32)
    y = givens_rotate(x, angles)
    assert torch.allclose(x.norm(dim=1), y.norm(dim=1), atol=2e-6, rtol=2e-6)
    assert y.shape == x.shape
