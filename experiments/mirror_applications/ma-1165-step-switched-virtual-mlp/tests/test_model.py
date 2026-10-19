import math
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import N,M,orthogonal,initial_virtual,decoder_step,VirtualLoop,make_teacher

def test_one_shared_vector_every_timestep():
    for mode in ('tied','same','switch','linear'):
        model=VirtualLoop(mode,4,seed=5)
        assert len(list(model.parameters()))==1
        assert model.learned_count()==M
        assert len(model.virtual())==4
        assert all(w.shape==(N,) for w in model.virtual())

def test_inverse_orthogonal_and_seed():
    torch.manual_seed(9)
    x=torch.randn(N,dtype=torch.float64)
    for t in (0,1,2,3):
        recovered=orthogonal(orthogonal(x,t),t,inverse=True)
        assert torch.allclose(recovered,x,rtol=0,atol=1e-12)
    assert torch.linalg.vector_norm(orthogonal(x,0,seed=True)).item()==pytest.approx(torch.linalg.vector_norm(x).item(),abs=1e-12)

def test_step_changes_are_real_and_not_only_a_gauge():
    z=initial_virtual(torch.randn(M)*.3)
    s0=decoder_step(z,0,'switch')
    s1=decoder_step(z,1,'switch')
    assert (s0-s1).abs().max()>1e-4
    f0=decoder_step(z,0,'same')
    f1=decoder_step(z,1,'same')
    assert torch.allclose(f0,f1)
    linear=decoder_step(z,2,'linear')
    assert torch.allclose(linear,(1+.16)*z, atol=1e-6,rtol=1e-6)

def test_mirror_is_nonlinear():
    x=torch.randn(M)*.25
    a,b=initial_virtual(x),initial_virtual(2*x)
    one=decoder_step(a,1,'switch')
    twice=decoder_step(b,1,'switch')
    assert not torch.allclose(twice,2*one,atol=1e-5,rtol=1e-5)

def test_step_specific_gradients():
    x=torch.randn(8,8)
    model=VirtualLoop('switch',4,seed=8)
    loss=(model(x)-x).square().mean()
    loss.backward()
    assert model.u.grad is not None and torch.isfinite(model.u.grad).all() and model.u.grad.norm()>0

def test_swap_changes_a_trained_mirror_trajectory():
    x=torch.randn(6,8)
    model=VirtualLoop('switch',4,seed=9)
    y=model(x)
    ys=model.swap_output(x,[3,2,1,0])
    assert (y-ys).abs().max()>1e-7
    assert model.swap_output(x,[0,1,2,3]).allclose(y)

def test_serialized_payload_accounting():
    m=VirtualLoop('switch',4)
    assert m.learned_count()==M
    assert m.payload_bytes()>=M*4
    assert VirtualLoop('untied',4).learned_count()==4*N
    assert VirtualLoop('gain',4).learned_count()==M+8

def test_teacher_is_deterministic_and_no_external_labels():
    x=torch.randn(7,8)
    a=make_teacher('aligned',4,19)(x)
    b=make_teacher('aligned',4,19)(x)
    c=make_teacher('independent',4,19)(x)
    assert torch.equal(a,b)
    assert (a-c).abs().max()>1e-5
