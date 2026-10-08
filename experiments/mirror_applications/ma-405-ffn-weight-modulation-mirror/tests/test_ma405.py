import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from run import rotation,style_matrix,world,D

def test_givens_is_orthogonal():
 r=rotation(torch.randn(4,D//2))
 assert torch.max(torch.abs(r.transpose(-1,-2)@r-torch.eye(D)))<1e-6

def test_style_demod_normalizes_rows():
 w=torch.randn(D,D);log_s=torch.randn(8,D)*.3
 m=style_matrix(w,log_s)
 assert torch.max(torch.abs(torch.linalg.vector_norm(m,dim=-1)-1))<1e-5

def test_world_has_separate_input_holdout():
 x,h,y,xt,yt,w1,w2,w,tm=world(40501,n=16)
 assert x.shape==xt.shape==(16,D)
 assert y.shape==yt.shape==(16,40,D)
 assert not torch.equal(x,xt)
