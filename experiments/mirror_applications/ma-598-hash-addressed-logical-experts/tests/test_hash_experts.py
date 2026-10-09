import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_hash_experts import METHODS,givens_rotate,hash_map


def test_hash_salts_are_deterministic_and_expert_maps_differ():
    a=hash_map(2048,8801);b=hash_map(2048,8801);c=hash_map(2048,8802)
    assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
    assert not np.array_equal(a[0],c[0])
    assert len(METHODS)==7


def test_expert_givens_are_orthogonal_views():
    torch.manual_seed(5);x=torch.randn(10,64);m=torch.randn(32)
    assert torch.allclose(x.norm(dim=1),givens_rotate(x,m).norm(dim=1),atol=2e-6,rtol=2e-6)
