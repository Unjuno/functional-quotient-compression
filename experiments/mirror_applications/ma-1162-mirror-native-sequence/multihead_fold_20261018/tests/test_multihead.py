import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import pytest
import torch
from multihead import SharedMultihead,rotate_pairs,code_pairwise_kl

torch.set_num_threads(1)


def test_rotation_inverse():
    torch.manual_seed(12)
    x=torch.randn(3,4,8)
    a=torch.tensor(.5)
    assert torch.allclose(rotate_pairs(rotate_pairs(x,a),-a),x,atol=1e-6)


def test_qk_gauge_null():
    torch.manual_seed(13)
    q,k=torch.randn(2,6,8),torch.randn(2,6,8)
    a=torch.tensor(.63)
    lhs=rotate_pairs(q,a)@rotate_pairs(k,a).transpose(-1,-2)
    rhs=q@k.transpose(-1,-2)
    assert torch.allclose(lhs,rhs,atol=5e-6)


def test_one_kv_multihead_folded_equivalence():
    torch.manual_seed(14)
    m=SharedMultihead(heads=4)
    with torch.no_grad():
        m.q_code.copy_(torch.tensor([-.9,-.3,.4,.8]))
        m.v_code.copy_(torch.tensor([-.7,.8,-.3,.4]))
    x=torch.randn(2,13,16)
    q0=x[:,-1:]@m.Wq
    k0,v0=x@m.Wk,x@m.Wv
    e=m.decode_materialized(q0,k0,v0)
    lazy=m.decode_lazy(q0,k0,v0)
    fold=m.decode_folded_native(x[:,-1:],k0,v0)
    assert torch.allclose(e,lazy,rtol=1e-5,atol=2e-6)
    assert torch.allclose(e,fold,rtol=1e-5,atol=2e-6)
    assert m.canonical_bytes(13,2)==2*2*13*8*4


def test_heads_are_distinct_with_relative_rotations():
    torch.manual_seed(15)
    m=SharedMultihead(heads=4)
    with torch.no_grad():m.q_code.copy_(torch.tensor([-.9,-.3,.4,.8]))
    _,a=m(torch.randn(3,12,16),return_attention=True)
    assert code_pairwise_kl(a)>1e-6


def test_null_heads_have_no_distinct_attention():
    torch.manual_seed(15)
    m=SharedMultihead(heads=4,mode='null')
    _,a=m(torch.randn(3,12,16),return_attention=True)
    assert code_pairwise_kl(a)==0


def test_nonlinear_views_not_linear_foldable():
    torch.manual_seed(16)
    m=SharedMultihead(heads=2,mode='nonlinear')
    with torch.no_grad():
        m.q_code.copy_(torch.tensor([-.6,.75]));m.v_code.copy_(torch.tensor([.4,-.65]))
    q=torch.randn(2,1,8)
    k,v=torch.randn(2,10,8),torch.randn(2,10,8)
    o=m.decode_lazy(q,k,v)
    assert torch.isfinite(o).all()
    a=torch.tensor(.71)
    v_pre=rotate_pairs(torch.tanh(rotate_pairs(v,a)),-a)
    from multihead import attention
    wrong=attention(q,k,v_pre)
    correct=rotate_pairs(torch.tanh(rotate_pairs(attention(q,k,v),a)),-a)
    assert (wrong-correct).abs().max()>1e-3


def test_all_heads_causal():
    torch.manual_seed(17)
    for mode in ['linear','nonlinear','gain','null']:
        m=SharedMultihead(heads=4,mode=mode)
        x=torch.randn(2,13,16)
        y=x.clone();y[:,9:]+=torch.randn_like(y[:,9:])*5
        a,b=m(x),m(y)
        assert torch.equal(a[:,:9],b[:,:9]),mode


def test_angle_gradients_are_finite_and_nonzero():
    torch.manual_seed(18)
    for mode in ['linear','nonlinear','gain']:
        m=SharedMultihead(heads=4,mode=mode)
        with torch.no_grad():m.q_code.copy_(torch.tensor([-.5,-.2,.3,.7]));m.v_code.copy_(torch.tensor([-.6,.1,.4,.8]))
        y=m(torch.randn(5,8,16))
        loss=(y*torch.randn_like(y)).mean();loss.backward()
        for p in (m.q_code,m.v_code):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>1e-8


def test_serialization_roundtrip():
    torch.manual_seed(19)
    m=SharedMultihead(heads=4)
    x=torch.randn(2,7,16)
    import io
    b=io.BytesIO()
    torch.save({'state':m.state_dict()},b)
    copy=SharedMultihead(heads=4)
    copy.load_state_dict(torch.load(io.BytesIO(b.getvalue()),weights_only=True)['state'])
    assert torch.equal(m(x),copy(x))
    assert m.serialized_bytes()>0
