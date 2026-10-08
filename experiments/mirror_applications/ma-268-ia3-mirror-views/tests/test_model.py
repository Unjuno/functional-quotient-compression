import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import TaskNet,METHODS,T,D,O,rotate4

def test_shapes_serialization():
 x=torch.randn(8,D);t=torch.arange(8)%T
 for method in METHODS:
  m=TaskNet(method);y=m(x,t);assert y.shape==(8,O) and torch.isfinite(y).all() and m.payload_bytes()>0

def test_rotation_norm():
 x=torch.randn(5,16);a=torch.randn(5,4);assert torch.allclose(x.norm(dim=-1),rotate4(x,a).norm(dim=-1),atol=1e-5)
