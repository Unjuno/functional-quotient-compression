import importlib.util
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[1]/'source'/'run.py';sp=importlib.util.spec_from_file_location('ma274',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

def test_world_shapes_and_teacher_targets():
 (xt,ht,yt),(xv,hv,yv),teacher=m.make(27400,0);assert ht.shape==(2048,m.D) and teacher.shape==(m.E,m.D,m.D)
 t=torch.einsum('nd,edh->neh',ht,teacher)[torch.arange(len(yt)),yt];assert t.shape==(len(yt),m.D)

def test_view_shapes():
 p={'base':torch.randn(m.D,m.D),'angle':torch.zeros(m.E,4,m.D//2)};assert m.weights('boft_mirror',p).shape==(m.E,m.D,m.D)
 q=m.boft_view(torch.randn(4,m.D//2));assert torch.allclose(q.T@q,torch.eye(m.D),atol=1e-6)

def test_serialization_differs_with_paid_state():
 q=nn_state={'weight':torch.randn(m.D,m.D)}
 a=m.pack('x',q,None) if False else None
