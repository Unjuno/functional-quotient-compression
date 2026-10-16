import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'source'))
from model import DualSwiGLU, generate_world, serialize_inference, infer_serialized, run_model


def test_exact_zero_rotation_is_shared_swiglu():
    torch.manual_seed(17)
    x=torch.randn(7,8)
    m=DualSwiGLU('mirror', d=8,h=24,K=3,q=12)
    m.set_active(12)
    y=m(x)
    shared=(torch.nn.functional.silu(x@m.wg)*(x@m.wu))@m.wd
    assert torch.allclose(y,shared[:,None,:].expand(-1,9,-1),atol=1e-6)


def test_pair_axes_are_compositional():
    torch.manual_seed(18)
    x=torch.randn(9,8)
    m=DualSwiGLU('mirror',d=8,h=24,K=3,q=12)
    with torch.no_grad():
        m.gangle[0,0]=0.4
        m.uangle[2,1]=-0.75
    y=m(x)
    assert not torch.allclose(y[:,0],y[:,6])
    assert not torch.allclose(y[:,0],y[:,2])
    assert torch.allclose(y[:,3],y[:,4],atol=1e-6)  # same gate 1, up 0/1 unused


def test_same_parameter_count_gain():
    for q in [1,4,12]:
        m=DualSwiGLU('mirror',d=8,h=24,K=3,q=q)
        g=DualSwiGLU('gain',d=8,h=24,K=3,q=q)
        assert m.active_trainable_params()==g.active_trainable_params()


def test_function_preserving_expansion_and_nonzero_new_gradient():
    torch.manual_seed(3)
    x=torch.randn(31,8)
    m=DualSwiGLU('mirror',d=8,h=24,K=3,q=12)
    m.set_active(1)
    before=m(x).detach().clone()
    m.set_active(4)
    assert torch.equal(m(x),before)
    loss=(m(x).square()*torch.arange(9).view(1,9,1)).mean()
    loss.backward()
    assert m.gangle.grad[:,1:4].abs().sum()>1e-8
    assert m.uangle.grad[:,1:4].abs().sum()>1e-8


def test_independent_upper_control_shape():
    x=torch.randn(5,8)
    for name in ['tied','mirror','gain','mirror_gate','mirror_up','pair_mirror','pair_gain','factor_lora1','independent']:
        m=DualSwiGLU(name,d=8,h=24,K=3,q=4)
        y=m(x)
        assert y.shape==(5,9,8) and torch.isfinite(y).all(),name


def test_serialized_reconstruction_and_capacity_bytes():
    x=torch.randn(7,8)
    m=DualSwiGLU('mirror',d=8,h=24,K=3,q=12)
    m.set_active(4)
    by=serialize_inference(m)
    rec=infer_serialized(by,x)
    assert torch.allclose(rec,m(x),atol=1e-6)
    assert len(by)<len(serialize_inference(DualSwiGLU('independent',d=8,h=24,K=3,q=12)))


def test_teacher_data_splits_and_heldout_pairs():
    w=generate_world(8001,'aligned',d=8,h=24,K=3)
    assert w['train'][0].shape==(512,8)
    assert w['test'][1].shape==(512,9,8)
    assert sorted(w['heldout_pairs'])==[2,3,7]
    for a in ['train','val','test']:
        assert torch.isfinite(w[a][1]).all()
    assert not torch.equal(w['train'][0],w['test'][0])


def test_pairwise_mirror_does_not_equal_factorized_capacity():
    a=DualSwiGLU('pair_mirror',d=8,h=24,K=3,q=4)
    b=DualSwiGLU('mirror',d=8,h=24,K=3,q=4)
    assert a.active_trainable_params()>b.active_trainable_params()
