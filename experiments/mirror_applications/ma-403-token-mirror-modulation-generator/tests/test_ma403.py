import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from run import rot,world

def test_rotation_is_orthogonal():
 a=torch.randn(5,8);r=rot(a)
 assert torch.max(torch.abs(r.transpose(-1,-2)@r-torch.eye(16)))<1e-6

def test_world_splits_by_token_position():
 _,_,p,_,_,_,pt,_,_,_=world(40301,n=64)
 assert float(p.max())<.8
 assert float(pt.min())>=.8 and float(pt.max())<1.0
