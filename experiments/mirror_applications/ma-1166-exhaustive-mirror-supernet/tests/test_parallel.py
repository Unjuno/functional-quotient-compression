import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import torch
import pytest
from phase_c_model import VirtualSupernet, NativeMLP
from parallel_model import ParallelGraph

@pytest.mark.parametrize('decoder',['mirror','affine'])
def test_serial_batched_matches_reference_forward_and_gradients(decoder):
    a=VirtualSupernet(decoder,1977)
    b=ParallelGraph(decoder,'serial_batched',1977)
    b.load_state_dict(a.state_dict())
    x1=torch.randn(10,6,requires_grad=True);x2=x1.detach().clone().requires_grad_(True)
    y1=a(x1);y2=b(x2)
    torch.testing.assert_close(y1,y2,atol=2e-6,rtol=2e-6)
    l1=y1.square().sum();l2=y2.square().sum()
    l1.backward();l2.backward()
    torch.testing.assert_close(a.theta.grad,b.theta.grad,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(x1.grad,x2.grad,atol=2e-6,rtol=2e-5)

@pytest.mark.parametrize('decoder',['mirror','affine'])
@pytest.mark.parametrize('steps',[2,6])
def test_sync_vectorized_matches_reference(decoder,steps):
    a=ParallelGraph(decoder,'sync',1977,steps=steps)
    x=torch.randn(8,6)
    y=a(x)
    yref=a.reference_sync(x)
    torch.testing.assert_close(y,yref,atol=3e-6,rtol=5e-6)
    assert y.shape==(8,6)

@pytest.mark.parametrize('mode,steps',[('serial_batched',1),('sync',2),('sync',6)])
def test_all_candidate_edges_and_gradients(mode,steps):
    a=ParallelGraph('mirror',mode,1977,steps=steps)
    x=torch.randn(7,6)
    y,st=a(x,return_stats=True)
    assert st['candidate_edges']==63
    assert st['active_edge_evals']==63*(steps if mode=='sync' else 1)
    assert sum(p.numel() for p in a.parameters())==253
    y.square().mean().backward()
    assert torch.isfinite(a.theta.grad).all()
    assert (a.theta.grad[:64].abs()>0).any()
    assert (a.theta.grad[64:127].abs()>0).any()
    assert (a.theta.grad[127:].abs()>0).any()

def test_skip_connection_and_shared_parameter_identity():
    net=ParallelGraph('mirror','sync',1977,steps=2)
    assert (0,6,0) in net.spec.edges and (1,6,2) in net.spec.edges
    assert len(list(net.parameters()))==1
    assert tuple(net.theta.shape)==(253,)

def test_native_budget():
    assert sum(p.numel() for p in NativeMLP().parameters())==253

@pytest.mark.parametrize('decoder',['mirror','affine'])
@pytest.mark.parametrize('steps',[2,6])
def test_sync_vectorized_matches_reference_gradients(decoder,steps):
    net=ParallelGraph(decoder,'sync',1977,steps=steps)
    x=torch.randn(3,6)
    p=(net(x)**2).sum()
    g=torch.autograd.grad(p,net.theta)[0]
    q=(net.reference_sync(x)**2).sum()
    h=torch.autograd.grad(q,net.theta)[0]
    torch.testing.assert_close(g,h,atol=3e-5,rtol=3e-5)
