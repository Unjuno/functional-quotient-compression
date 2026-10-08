import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import Experts,METHODS,E,D,O

def test_all_methods_shape_finite_and_payload():
 x=torch.randn(12,D);e=torch.arange(12)%E
 for method in METHODS:
  m=Experts(method);y=m(x,e)
  assert y.shape==(12,O) and torch.isfinite(y).all() and m.payload_bytes()>0

def test_rotate_preserves_norm():
 from model import rotate
 x=torch.randn(10,8);a=torch.tensor(.32)
 assert torch.allclose(x.norm(dim=-1),rotate(x,a).norm(dim=-1),atol=1e-6)
