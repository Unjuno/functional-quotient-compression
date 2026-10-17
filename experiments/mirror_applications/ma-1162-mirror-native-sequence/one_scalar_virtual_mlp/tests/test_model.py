import io
import math
import torch
from source.model import VirtualMLP, DenseMLP, WEIGHTS, mirror

torch.set_num_threads(1)


def test_exactly_one_trainable_scalar_for_all_modes():
    for kind in ('mirror','spectral','linear_view','scalar_gain'):
        m=VirtualMLP(kind)
        assert list(dict(m.named_parameters()).keys())==['s']
        assert sum(p.numel() for p in m.parameters())==1
        assert len(m.state_dict())==1


def test_actual_virtual_mlp_has_hundreds_of_weights():
    m=VirtualMLP()
    weights=m.virtual_weights()
    assert [(tuple(w.shape),tuple(b.shape)) for w,b in weights]==[
        ((16,2),(16,)),((16,16),(16,)),((1,16),(1,))]
    assert WEIGHTS==337
    assert sum(w.numel()+b.numel() for w,b in weights)==337


def test_mirror_inverse_nonlinear_nontrivial():
    m=VirtualMLP()
    v=torch.arange(20,dtype=torch.float64)*.23
    a=torch.tensor(.61,dtype=torch.float64)
    assert torch.allclose(mirror(mirror(v,a,7),a,7,True),v,atol=1e-13,rtol=1e-13)
    assert not torch.allclose(m.virtual_vector(),VirtualMLP('spectral').virtual_vector())
    assert not torch.allclose(m.virtual_vector(),VirtualMLP('linear_view').virtual_vector(), atol=1e-9)


def test_linear_mirror_is_gauge_null():
    a=VirtualMLP('linear_view').virtual_vector()
    b=VirtualMLP('spectral').virtual_vector()
    assert torch.allclose(a,b,rtol=0,atol=1e-14)


def test_autograd_gradient_finite_difference():
    model=VirtualMLP('mirror',init_s=.34)
    x=torch.randn((17,2),dtype=torch.float64)
    loss=(model(x).square()).mean()
    loss.backward()
    g=model.s.grad.item()
    assert math.isfinite(g) and abs(g)>1e-9
    eps=1e-6
    with torch.no_grad():
        model.s.add_(eps)
        plus=model(x).square().mean().item()
        model.s.add_(-2*eps)
        minus=model(x).square().mean().item()
        model.s.add_(eps)
    num=(plus-minus)/(2*eps)
    assert math.isclose(g,num,rel_tol=2e-5,abs_tol=1e-9),(g,num)


def test_one_scalar_changes_all_three_virtual_matrix_roles():
    m=VirtualMLP(init_s=.2)
    w0=[w.detach().clone() for w,b in m.virtual_weights()]
    with torch.no_grad():m.s.add_(.1)
    w1=[w.detach().clone() for w,b in m.virtual_weights()]
    assert all((a-b).abs().max()>.001 for a,b in zip(w0,w1))


def test_no_hidden_second_parameter_in_optimizer():
    m=VirtualMLP()
    opt=torch.optim.Adam([m.s],lr=.05)
    assert len(opt.param_groups)==1 and len(opt.param_groups[0]['params'])==1
    x=torch.randn(32,2,dtype=torch.float64)
    target=torch.zeros(32,1,dtype=torch.float64)
    s0=m.s.item()
    for _ in range(4):
        opt.zero_grad();loss=(m(x)-target).square().mean();loss.backward();opt.step()
    assert s0 != m.s.item()


def test_roundtrip_exact_and_bytes():
    m=VirtualMLP(init_s=-.39)
    x=torch.randn(5,2,dtype=torch.float64)
    f=io.BytesIO();torch.save({'architecture':m.metadata(),'state':m.state_dict()},f)
    clone=VirtualMLP(mode=m.mode,rounds=m.rounds,dtype=m.s.dtype)
    clone.load_state_dict(torch.load(io.BytesIO(f.getvalue()),weights_only=True)['state'])
    assert torch.equal(m(x),clone(x))
    assert len(f.getvalue())==m.checkpoint_bytes()


def test_dense_same_architecture_but_more_parameters():
    d=DenseMLP()
    assert sum(p.numel() for p in d.parameters())==337
    assert d(torch.randn(13,2,dtype=torch.float64)).shape==(13,1)


def test_all_model_outputs_are_finite():
    x=torch.randn(30,2,dtype=torch.float64)*3
    for k in ('mirror','spectral','linear_view','scalar_gain'):
        y=VirtualMLP(k)(x)
        assert torch.isfinite(y).all() and y.shape==(30,1)
