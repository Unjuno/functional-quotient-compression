import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import TaskFamilyModel,METHODS,T,D,O,givens

def test_shapes_finite_payload():
 x=torch.randn(7,D);task=torch.arange(7)%T
 for method in METHODS:
  m=TaskFamilyModel(method,context_seed=123);y=m(x,task)
  assert y.shape==(7,O) and torch.isfinite(y).all() and m.payload_bytes()>0

def test_orthogonal_view_preserves_norm():
 z=torch.randn(4,O);angles=torch.randn(4,8)
 y=givens(z,angles)
 assert torch.allclose(y.norm(dim=-1),z.norm(dim=-1),atol=1e-5)
