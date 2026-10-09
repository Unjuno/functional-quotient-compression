import sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_head_distill import METHODS,make_world,ptrain,maps,forward,H,D,HD

def test_world_is_deterministic():
 a=make_world(60101);b=make_world(60101)
 assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])

def test_independent_students_do_not_copy_teacher():
 x,y,xt,yt,q,k,v,wo=make_world(60101);p=ptrain('independent_heads',60101,(q,k,v,wo))
 assert not torch.equal(p['q'],torch.tensor(q))
 assert p['q'].shape==(H,D,HD)

def test_gqa_mqa_and_mirror_shapes():
 w=make_world(60101);teacher=w[4:8];x=torch.tensor(w[0][:2])
 for m in ('gqa2','mqa','mirror_givens_hash'):
  p=ptrain(m,60101,teacher);out,ctx,att=forward(x,p,m,maps(m,60101))
  assert out.shape==(2,8,32) and ctx.shape==(2,H,8,HD) and att.shape==(2,H,8,8)
 assert METHODS[-1]=='rank1_hash'
