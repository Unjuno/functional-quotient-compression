import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import pytest


def api():
    import engine
    return engine


def test_rotate_inverse():
    e=api(); x=torch.randn(3,7,16,dtype=torch.float64); a=torch.randn(3,8,dtype=torch.float64)
    assert torch.allclose(e.rotate(e.rotate(x,a),-a),x,atol=1e-12)


def test_zero_mirror_is_dense():
    e=api(); m=e.Bank('mirror',8,[0]); m.codes.data.zero_(); x=torch.randn(13,5); c=torch.randn(13,6)
    pred,p,z=m(x,c); dense=torch.nn.functional.gelu(x@m.w1[0]+m.b1[0])@m.w2[0]+m.b2[0]
    assert torch.allclose(pred[0],dense,atol=1e-6)
    assert torch.equal(z[:,:,0],z[:,:,1])


def test_capacity_nested_identity_extension():
    e=api(); a=e.Bank('mirror',1,[4]); b=e.Bank('mirror',8,[4]); b.codes.data.zero_(); b.codes.data[:,:,:1]=a.codes.data
    x=torch.randn(7,5);c=torch.randn(7,6)
    assert torch.allclose(a(x,c)[0],b(x,c)[0],atol=1e-6)


def test_native_gain_parameter_match():
    e=api()
    for q in [1,4,8]:
        a=e.Bank('mirror',q,[2]);b=e.Bank('gain',q,[2])
        assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())


def test_zero_route_and_warm_are_half():
    e=api();m=e.Bank('mirror',8,[0,1],zero_route=True);x=torch.randn(9,5);c=torch.randn(9,6)
    assert torch.equal(m(x,c)[1],torch.full((2,9),.5))
    m.router.data.add_(3)
    assert torch.equal(m(x,c,uniform=True)[1],torch.full((2,9),.5))


def test_expert_swap_invariance():
    e=api();a=e.Bank('mirror',8,[1]);b=e.Bank('mirror',8,[1]);b.codes.data=a.codes.data.flip(1);b.router.data.neg_();b.rb.data.neg_()
    x=torch.randn(9,5);c=torch.randn(9,6)
    assert torch.allclose(a(x,c)[0],b(x,c)[0],atol=1e-6)


def test_replica_step_independence():
    e=api();torch.set_num_threads(1)
    a=e.Bank('mirror',4,[0,1]);b=e.Bank('mirror',4,[1]);x=torch.randn(13,5);c=torch.randn(13,6);y=torch.randn(13)
    oa=torch.optim.Adam(a.parameters(),lr=.01,foreach=False);ob=torch.optim.Adam(b.parameters(),lr=.01,foreach=False)
    for _ in range(3):
        for m,o in [(a,oa),(b,ob)]:
            o.zero_grad();((m(x,c)[0]-y)**2).mean(1).sum().backward();o.step()
    for p,q in zip(a.parameters(),b.parameters()): assert torch.allclose(p[1],q[0],atol=2e-6,rtol=1e-5)


def test_deterministic_data_and_splits():
    e=api();a=e.world(97200,'aligned');b=e.world(97200,'aligned')
    for k in a: assert np.array_equal(a[k],b[k])
    assert len(set(map(tuple,a['train_x'])) & set(map(tuple,a['test_x'])))==0
    assert a['train_y'].shape==(512,)
    assert not np.array_equal(a['train_y'],a['train_clean'])


def test_codec_roundtrip(tmp_path):
    e=api();m=e.Bank('mirror',4,[0,1]);p=tmp_path/'model.bin';h=e.save(m,1,p);m2=e.load(p)
    x=torch.randn(7,5);c=torch.randn(7,6)
    assert torch.equal(m(x,c)[0][1],m2(x,c)[0][0])
    assert e.save(m2,0,tmp_path/'again.bin')==h


def test_no_zero_code_initialization():
    e=api();m=e.Bank('mirror',8,[0,1]);assert (m.codes.abs().sum((1,2))>0).all()


def test_oracle_not_default_route():
    e=api();m=e.Bank('mirror',8,[1]);x=torch.randn(9,5);c=torch.randn(9,6);reg=torch.arange(9)%2
    _,p,_=m(x,c,oracle=reg)
    assert torch.equal(p[0],reg.float())
    assert not torch.equal(m(x,c)[1][0],reg.float())


def test_gradient_contributions_sum_to_shared_gradient():
    e=api();m=e.Bank('mirror',8,[0,1]);x=torch.randn(13,5);c=torch.randn(13,6);y=torch.randn(13)
    pred,p,z=m(x,c);params=[m.w1,m.b1,m.w2,m.b2]
    total=torch.autograd.grad((pred-y).square().mean(1).sum(),params,retain_graph=True)
    upstream=(2*(pred-y)/len(x)).detach();parts=[]
    for i,w in [(0,1-p),(1,p)]:
        parts.append(torch.autograd.grad((upstream*w.detach()*z[:,:,i]).sum(),params,retain_graph=True))
    for g,a,b in zip(total,parts[0],parts[1]):assert torch.allclose(g,a+b,atol=1e-6)
    diag=e.gradient_diagnostic(m,x,c,y)
    assert all(-1.000001<=z<=1.000001 for z in diag['cos'])


def test_code_gradient_finite_difference():
    e=api();m=e.Bank('mirror',4,[1]).double();x=torch.randn(5,5,dtype=torch.float64);c=torch.randn(5,6,dtype=torch.float64)
    val=m(x,c)[0].square().sum();grad=torch.autograd.grad(val,m.codes)[0][0,1,2].item()
    orig=m.codes.data[0,1,2].item();eps=1e-5
    m.codes.data[0,1,2]=orig+eps;a=m(x,c)[0].square().sum().item()
    m.codes.data[0,1,2]=orig-eps;b=m(x,c)[0].square().sum().item()
    assert abs(grad-(a-b)/(2*eps))<1e-8


def test_calibration_uses_validation_only():
    import runner
    rows=[{'lr':.01,'val_noisy_mse':2.,'validation_scale':1.,'test_mse':0.},
          {'lr':.003,'val_noisy_mse':1.,'validation_scale':1.,'test_mse':1e10}]
    assert runner.choose_lr(rows)==.003
