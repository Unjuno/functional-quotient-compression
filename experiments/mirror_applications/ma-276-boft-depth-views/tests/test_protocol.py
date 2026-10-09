import importlib.util
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[1]/'source'/'run.py';sp=importlib.util.spec_from_file_location('ma276',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

def test_rotation_and_recurrence():
 q=m.rot(torch.tensor(.33));assert torch.allclose(q.T@q,torch.eye(m.D),atol=1e-6)
 x=torch.randn(5,m.D);assert m.recur(x,[torch.zeros(m.D,m.D)]*m.K).shape==x.shape

def test_world_shapes():
 x,xv,b,a,t,ind=m.make(27600,0);assert x.shape==(1024,m.D) and t.shape==ind.shape==(m.K,m.D,m.D)

def test_lora_decode():
 p={'base':torch.randn(m.D,m.D),'A':torch.randn(m.K,m.D,2),'B':torch.randn(m.K,2,m.D)};assert len(m.decode('lora',p))==m.K
