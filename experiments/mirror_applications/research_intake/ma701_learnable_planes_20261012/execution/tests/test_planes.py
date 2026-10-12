import pytest
import torch
from source.planes import orthonormal_planes, apply_planes, householder_alias

def fixture():
    g=torch.Generator().manual_seed(31)
    a=torch.randn(3,8,generator=g,dtype=torch.float64)
    b=torch.randn(3,8,generator=g,dtype=torch.float64)
    x=torch.randn(7,2,8,generator=g,dtype=torch.float64)
    m=torch.randn(2,3,generator=g,dtype=torch.float64)*.4
    return x,a,b,m

def test_orthonormal_planes():
    x,a,b,m=fixture();u,v=orthonormal_planes(a,b)
    assert torch.allclose(u.norm(dim=-1),torch.ones(3,dtype=u.dtype),atol=1e-12)
    assert torch.allclose(v.norm(dim=-1),torch.ones(3,dtype=u.dtype),atol=1e-12)
    assert (u*v).sum(-1).abs().max()<1e-12

def test_zero_angles_identity():
    x,a,b,m=fixture(); assert torch.equal(apply_planes(x,a,b,torch.zeros_like(m)),x)

def test_norm_and_orthogonality():
    x,a,b,m=fixture(); y=apply_planes(x,a,b,m)
    assert torch.allclose(x.square().sum(-1),y.square().sum(-1),atol=1e-11)
    eye=torch.eye(8,dtype=a.dtype).unsqueeze(1).expand(-1,2,-1)
    q=apply_planes(eye,a,b,m).transpose(0,1)
    assert torch.allclose(q.transpose(-1,-2)@q,torch.eye(8,dtype=a.dtype),atol=1e-11)

def test_native_two_reflection_alias():
    x,a,b,m=fixture()
    assert torch.allclose(apply_planes(x,a,b,m),householder_alias(x,a,b,m),atol=1e-12)

def test_gradcheck():
    x,a,b,m=fixture()
    assert torch.autograd.gradcheck(apply_planes,(x.requires_grad_(),a.requires_grad_(),b.requires_grad_(),m.requires_grad_()),fast_mode=True,atol=2e-5)

def test_zero_angle_blocks_axis_gradient():
    x,a,b,m=fixture();a.requires_grad_();b.requires_grad_();m=torch.zeros_like(m,requires_grad=True)
    apply_planes(x,a,b,m)[...,0].sum().backward()
    assert a.grad.abs().max()==0 and b.grad.abs().max()==0
    assert m.grad.abs().max()>0

def test_nonzero_angle_task_gradient_reaches_axes():
    x,a,b,m=fixture();a.requires_grad_();b.requires_grad_();m.requires_grad_()
    apply_planes(x,a,b,m)[...,0].sum().backward()
    for v in (a,b,m): assert torch.isfinite(v.grad).all() and v.grad.norm()>1e-7

def test_degenerate_plane_rejected():
    with pytest.raises(ValueError):orthonormal_planes(torch.zeros(1,8),torch.ones(1,8))
    with pytest.raises(ValueError):orthonormal_planes(torch.ones(1,8),torch.ones(1,8))

def test_noncommuting_order():
    x,a,b,m=fixture()
    assert not torch.allclose(apply_planes(x,a,b,m),apply_planes(x,a.flip(0),b.flip(0),m.flip(-1)),atol=1e-5)
