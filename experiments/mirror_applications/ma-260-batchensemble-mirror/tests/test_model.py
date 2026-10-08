import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import Ensemble,METHODS,D,C,M

def test_shapes_and_serialization():
 x=torch.randn(9,D)
 for method in METHODS:
  m=Ensemble(method,31);z=m.forward_members(x)
  assert z.shape==(9,M,C) and torch.isfinite(z).all() and m.payload_bytes()>0

def test_shared_member_replication():
 m=Ensemble('shared');z=m.forward_members(torch.randn(3,D));assert torch.equal(z[:,0],z[:,1])
