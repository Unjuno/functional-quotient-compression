import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import RoutedFFN,METHODS,E,D,butterfly

def test_methods_shape_payload():
 x=torch.randn(12,D);e=torch.arange(12)%E
 for method in METHODS:
  m=RoutedFFN(method);y=m(x,e);assert y.shape==(12,D) and torch.isfinite(y).all() and m.payload_bytes()>0

def test_butterfly_is_orthogonal_on_last_axis():
 x=torch.randn(5,D);a=torch.randn(4,8);y=butterfly(x,a);assert torch.allclose(x.norm(dim=-1),y.norm(dim=-1),atol=1e-5)
