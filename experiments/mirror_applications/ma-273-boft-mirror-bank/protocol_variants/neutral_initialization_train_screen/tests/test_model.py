import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import Bank,METHODS,T,D,butterfly

def test_shapes_orthogonality_and_bytes():
 x=torch.randn(6,D);a=torch.randn(6,4,8)
 assert torch.allclose(butterfly(x,a).norm(dim=-1),x.norm(dim=-1),atol=1e-5)
 for k in METHODS:
  m=Bank(k);z=m(x,torch.arange(6)%T);assert z.shape==(6,D) and torch.isfinite(z).all() and m.payload_bytes()>0
