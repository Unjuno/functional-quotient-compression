import math
import torch
import pytest
from source.model import *

torch.set_num_threads(1)

@pytest.mark.parametrize('r,k',[(0,0),(1,2),(2,1),(11,1)])
def test_orthogonal_invertible(r,k):
    torch.manual_seed(501)
    x=torch.randn(N,dtype=torch.float64)
    assert torch.max(torch.abs(mirror_q(mirror_q(x,r,k),r,k,True)))<100
    assert torch.allclose(mirror_q(mirror_q(x,r,k),r,k,True),x,atol=2e-13)

def test_seed_inverse():
    torch.manual_seed(501)
    x=torch.randn(N,dtype=torch.float64)
    assert torch.allclose(seed_transform(seed_transform(x),True),x,atol=1e-12)

def test_exact_weight_and_output_conversion():
    torch.manual_seed(42)
    teacher=NativeMLP().eval()
    student=MirrorMLP(N,True).eval()
    compile_exact(teacher,student)
    w=weight_to_flat(teacher)
    assert (student.weights()-w).abs().max()<1e-10
    x=torch.randn(41,64,dtype=torch.float64)
    assert (teacher(x)-student(x)).abs().max()<1e-10

def test_program_nontrivial_and_trainable():
    student=MirrorMLP(16,True)
    student.coords.data.copy_(torch.linspace(-.2,.3,16))
    plain=MirrorMLP(16,False)
    plain.coords.data.copy_(student.coords)
    assert (student.weights()-plain.weights()).abs().mean()>1e-3
    (student.weights().square().sum()).backward()
    assert student.coords.grad is not None and torch.isfinite(student.coords.grad).all()
    assert student.coords.grad.abs().sum()>1e-7

def test_single_weight_tensor_trainable_only():
    for m in (1,16,64,256,N):
        a=MirrorMLP(m)
        p=list(a.parameters())
        assert len(p)==1 and p[0].numel()==m
        assert a.weights().shape==(N,)

def test_input_non_leakage():
    a=NativeMLP(); b=MirrorMLP(64)
    x=torch.rand(7,64,dtype=torch.float64)
    assert a(x).shape==b(x).shape==(7,10)
