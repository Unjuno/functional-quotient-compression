import importlib.util
from pathlib import Path
import torch

SCRIPT=Path(__file__).parents[1]/'source'/'dune_screen.py'
spec=importlib.util.spec_from_file_location('ma1009_dune',SCRIPT)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_multicoil_forward_adjoint_inner_product():
    q=mod.make_case(1009123,'B')
    x=torch.randn_like(q['zf']); y=torch.randn_like(q['kspace'])
    lhs=torch.sum(mod.forward(x,q['sens'],q['mask']).conj()*y).real
    rhs=torch.sum(x*mod.adjoint(y*q['mask'],q['sens']))
    assert torch.allclose(lhs,rhs,atol=2e-5,rtol=2e-5)

def test_final_projection_hits_acquired_samples():
    q=mod.make_case(1009124,'B');x=torch.randn_like(q['target'])
    out,kprojected=mod.exact_projection(x,q['kspace'],q['sens'],q['mask'])
    residual=((mod.forward(out,q['sens'],q['mask'])-q['kspace'])*q['mask']).norm()/q['kspace'].norm()
    assert float(residual)<1e-5
    assert out.shape==q['target'].shape

def test_decoder_vjp_matches_finite_difference():
    torch.manual_seed(1009125);m=mod.DUNE();z=torch.randn(1,4,8,8,requires_grad=True);g=torch.randn(1,2,8,8)
    x=m.decoder(z);vjp=torch.autograd.grad(x,z,grad_outputs=g)[0]
    direction=torch.randn_like(z);eps=1e-3
    lhs=((m.decoder(z+eps*direction)-m.decoder(z-eps*direction))*g).sum()/(2*eps)
    rhs=(vjp*direction).sum()
    assert torch.allclose(lhs,rhs,atol=2e-3,rtol=2e-3)

def test_latent_views_do_not_change_measurement_operator():
    q=mod.make_case(1009126,'B');m=mod.DUNE();m.mode='mirror';a=m(q,project=False)
    m.mirror_angle.data.fill_(.3);b=m(q,project=False)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.equal(q['mask'],mod.make_case(1009126,'B')['mask'])
    assert torch.equal(q['kspace'],mod.make_case(1009126,'B')['kspace'])

def test_prior_view_changes_unacquired_prediction_after_projection():
    q=mod.make_case(1009127,'B');m=mod.DUNE();m.mode='mirror'
    x0=m(q,project=False);m.mirror_angle.data.fill_(.4);x1=m(q,project=False)
    y0,k0=mod.exact_projection(x0,q['kspace'],q['sens'],q['mask']);y1,k1=mod.exact_projection(x1,q['kspace'],q['sens'],q['mask'])
    r0=mod.forward(y0,q['sens'],q['mask']);r1=mod.forward(y1,q['sens'],q['mask'])
    assert torch.allclose(r0[q['mask'].expand_as(r0)],q['kspace'][q['mask'].expand_as(r0)],atol=1e-5,rtol=1e-5)
    assert torch.allclose(r1[q['mask'].expand_as(r1)],q['kspace'][q['mask'].expand_as(r1)],atol=1e-5,rtol=1e-5)
    assert not torch.allclose(y0,y1)
