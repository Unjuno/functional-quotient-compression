import importlib.util
from pathlib import Path
import torch
P=Path(__file__).resolve().parents[1]/'source'/'run.py';sp=importlib.util.spec_from_file_location('ma360',P);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

def test_reversible_step_roundtrip():
    torch.manual_seed(1);x=torch.randn(7,m.D);f=torch.randn(m.H,m.H)*.1;g=torch.randn(m.H,m.H)*.1;s=torch.tensor(.4)
    y=m.RevStep.apply(x,f,g,s);a,b=y.chunk(2,-1);bb=b-s*torch.tanh(a@g);aa=a-s*torch.tanh(bb@f)
    assert torch.max(torch.abs(torch.cat([aa,bb],-1)-x))<1e-6

def test_reversible_gradient_matches_direct():
    torch.manual_seed(2);x=torch.randn(4,m.D,requires_grad=True);f=torch.randn(m.H,m.H,requires_grad=True)*.1;g=torch.randn(m.H,m.H,requires_grad=True)*.1;s=torch.tensor(.3,requires_grad=True)
    y=m.RevStep.apply(x,f,g,s);loss=(y*y).sum();gr=torch.autograd.grad(loss,(x,f,g,s))
    xx=x.detach().clone().requires_grad_();ff=f.detach().clone().requires_grad_();gg=g.detach().clone().requires_grad_();ss=s.detach().clone().requires_grad_();a,b=xx.chunk(2,-1);ap=a+ss*torch.tanh(b@ff);bp=b+ss*torch.tanh(ap@gg);yy=torch.cat([ap,bp],-1);gr2=torch.autograd.grad((yy*yy).sum(),(xx,ff,gg,ss))
    assert max(torch.max(torch.abs(a-b)).item() for a,b in zip(gr,gr2))<1e-5

def test_serialized_payload_deterministic():
    model=m.Coupling('tied',3);a=m.payload(model);b=m.payload(model);assert a==b
