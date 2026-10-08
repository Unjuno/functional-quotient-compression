import importlib.util
from pathlib import Path
import torch
P=Path(__file__).resolve().parents[1]/'source'/'run.py';sp=importlib.util.spec_from_file_location('ma361',P);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

def test_router_capacity_fixed():
    x=torch.randn(3,m.T,m.D);model=m.Net('tied_moD',1);_,mask=model(x);assert torch.all(mask.sum(1)==8)

def test_mirror_direct_same_function():
    x=torch.randn(2,m.T,m.D);a=m.Net('mirror_view',2);b=m.Net('direct_gate',2);b.load_state_dict(a.state_dict())
    y,_=a(x);z,_=b(x);assert torch.equal(y,z)

def test_archive_deterministic():
    x=torch.randn(1,m.T,m.D);a=m.Net('tied_moD',3);assert m.archive(a)==m.archive(a)
