import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import AdapterBank,METHODS,T,D,O

def test_shapes_and_payload():
 x=torch.randn(9,D);t=torch.arange(9)%T
 for method in METHODS:
  m=AdapterBank(method,77);y=m(x,t);assert y.shape==(9,O) and torch.isfinite(y).all() and m.payload_bytes()>0

def test_basis_exactly_reconstructs_from_seed():
 a=AdapterBank('vera',77);b=AdapterBank('vera',77);assert torch.equal(a.A,b.A) and torch.equal(a.B,b.B)
