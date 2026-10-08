import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import TaskViews,METHODS,T,D,O

def test_shapes_payload_and_orthogonality():
 x=torch.randn(7,D);t=torch.arange(7)%T
 for method in METHODS:
  m=TaskViews(method);y=m(x,t);assert y.shape==(7,O) and torch.isfinite(y).all() and m.payload_bytes()>0
  if method in ('oft','mirror'):
   q=m.mats();eye=torch.eye(O).expand(T,-1,-1);assert torch.allclose(q.transpose(-1,-2)@q,eye,atol=1e-5)
