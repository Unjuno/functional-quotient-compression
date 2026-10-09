import importlib.util
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[1]/'source'/'run.py';sp=importlib.util.spec_from_file_location('ma286',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

def test_world_shapes_and_aligned_factorization():
 w,x,xv,q,right,c,aligned,ind=m.make(28600,0);assert q.shape==(m.D,m.R) and right.shape==(m.R,m.D);assert aligned.shape==ind.shape==(m.T,m.D,m.D)
 for t in range(m.T):assert torch.allclose(aligned[t],q@torch.diag(c[t])@right)

def test_payload_counts_extra_state():
 w=torch.randn(m.D,m.D);assert len(m.pack([w],'x'))<len(m.pack([w,torch.randn(m.R,m.D)],'x'))
