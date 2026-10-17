"""Deterministic tests for the posthoc Mirror inference/byte accounting probe."""
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'source'))
sys.path.insert(0,str(ROOT/'source'))
from audit import mirror_pack,mirror_unpack,native_pack,native_unpack,HEADER_LEN,N,CoeffDecoder
from online_operator import hadamard,signs,direct,materialize,dense_forward,implicit_forward,direct_nonlinear_sandwich,dense_nonlinear_sandwich

def test_codec_roundtrip_quantized_weights_exact():
    torch.manual_seed(111)
    ix=torch.randperm(N)[:91].sort().values
    m=CoeffDecoder(ix,'mirror',2,1)
    with torch.no_grad():m.coords.copy_(torch.randn(91,dtype=torch.float64)*.1)
    p=mirror_pack(m.indices,m.coords,2,1)
    out=mirror_unpack(p)
    with torch.no_grad():
        assert torch.equal(out.coords,m.coords.half().double())
        expected=CoeffDecoder(ix,'mirror',2,1)
        expected.coords.copy_(m.coords.half().double())
        assert torch.equal(out.weights(),expected.weights())
    assert len(p)==HEADER_LEN+(N+7)//8+2*91

def test_native_same_dtype_roundtrip():
    torch.manual_seed(5)
    v=torch.randn(N,dtype=torch.float64)
    pay=native_pack(v)
    assert torch.equal(native_unpack(pay),v.half().double())
    assert len(pay)==HEADER_LEN+2*N

def test_codec_does_not_store_expanded_float_weight_tensor():
    torch.manual_seed(1)
    ix=torch.randperm(N)[:768].sort().values
    c=torch.randn(768,dtype=torch.float64)
    p=mirror_pack(ix,c,8,4)
    assert len(p)==1710
    assert len(p)<len(native_pack(torch.randn(N,dtype=torch.float64)))


def test_hadamard_involution():
    torch.manual_seed(8)
    x=torch.randn(4,128,dtype=torch.float64)
    assert torch.allclose(hadamard(hadamard(x)),x,atol=1e-12,rtol=0)


def test_direct_operator_matches_dense_weight():
    torch.manual_seed(6)
    d=64
    x=torch.randn(3,d,dtype=torch.float64)
    u=torch.randn(d,dtype=torch.float64)
    Q=hadamard(torch.eye(d,dtype=torch.float64))
    s=signs(3,7,d)
    W=materialize(u,s,Q)
    assert torch.allclose(direct(x,u,s),x@W.T,atol=1e-12,rtol=0)


def test_k_parallel_r_recurrence_implicit_equivalence():
    torch.manual_seed(14)
    d=64;k=4;r=3
    x=torch.randn(9,d,dtype=torch.float64)
    u=torch.randn(d,dtype=torch.float64)*.05
    Q=hadamard(torch.eye(d,dtype=torch.float64))
    s=[signs(i,j,d) for i in range(r) for j in range(k)]
    dense=torch.stack([materialize(u,z,Q) for z in s])
    a=implicit_forward(x,u,s,k,r)
    b=dense_forward(x,dense,k,r)
    assert torch.allclose(a,b,atol=1e-12,rtol=0)


def test_nonlinear_mirror_sandwich_matrix_free_and_gradient():
    torch.manual_seed(551)
    d=128
    x=torch.randn(7,d,dtype=torch.float64)
    u=torch.nn.Parameter(torch.randn(d,dtype=torch.float64)*.02)
    s=signs(4,2,d)
    h=hadamard(torch.eye(d,dtype=torch.float64))
    a=direct_nonlinear_sandwich(x,u,s)
    b=dense_nonlinear_sandwich(x,u,s,h)
    assert torch.allclose(a,b,atol=1e-12,rtol=0)
    a.square().sum().backward()
    assert u.grad is not None and torch.isfinite(u.grad).all() and u.grad.abs().sum()>1e-8
