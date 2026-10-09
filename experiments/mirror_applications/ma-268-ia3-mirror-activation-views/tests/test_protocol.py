import importlib.util
from pathlib import Path
import torch

P=Path(__file__).resolve().parents[1]/"source"/"run.py"
spec=importlib.util.spec_from_file_location("ma268",P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_rotation_and_shapes():
    x=torch.randn(12,m.H); y=m.rotate(x,torch.tensor(.43)); assert y.shape==x.shape
    assert torch.allclose(m.rotate(y,torch.tensor(-.43)),x,atol=1e-6)

def test_teacher_and_fit_oracle():
    w1,w2,xs,hs,xv,hv,angles,scales=m.make_world(26800,0);m.W2_CURRENT=w2
    target=m.rotate(hs,angles[3])@w2
    a,_=m.fit_angle(hs,target)
    assert abs(a-float(angles[3]))<1e-4
    assert m.nrmse(m.rotate(hv,torch.tensor(a))@w2,m.rotate(hv,angles[3])@w2)<1e-5

def test_payload_accounts_for_codes():
    shared=torch.randn(10);a=m.pack("v",shared,torch.zeros(1));b=m.pack("v",shared,torch.zeros(4))
    assert len(b)>len(a)
