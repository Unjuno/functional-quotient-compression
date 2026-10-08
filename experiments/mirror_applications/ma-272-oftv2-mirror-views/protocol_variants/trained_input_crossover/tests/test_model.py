import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import Views,METHODS,D,T,givens

def test_shape_and_payload():
 x=torch.randn(8,D);t=torch.arange(8)%T
 for k in METHODS:
  m=Views(k);y=m(x,t);assert y.shape==(8,D) and torch.isfinite(y).all() and m.payload_bytes()>0

def test_givens_norm():
 x=torch.randn(5,D);a=torch.randn(5,8);assert torch.allclose(x.norm(dim=-1),givens(x,a).norm(dim=-1),atol=1e-5)
