import hashlib
import pytest
import torch
from source.experiment import Model,make_world,fit,metrics,save_model,load_model,METHODS

def cfg():return dict(lr=.001,axis_lr=.001,warmup=0,sigma=.03,steps=3)

@pytest.mark.parametrize('family',['aligned','partial','unrelated','digits8','digits4'])
def test_world_shapes_and_firewall(family):
    w=make_world(19,family)
    assert len(set(w['train_ids'])&set(w['test_ids']))==0
    assert len(set(w['train_ids'])&set(w['val_ids']))==0
    assert not any('teacher' in k for k in w)
    assert torch.isfinite(w['x_train']).all() and torch.isfinite(w['y_train']).all()

def test_fresh_requires_explicit_unlock():
    with pytest.raises(ValueError):make_world(17111,'aligned')

@pytest.mark.parametrize('method',METHODS)
def test_methods_train_and_roundtrip(method,tmp_path):
    torch.set_num_threads(1)
    w=make_world(23,'digits8');m=Model(method,32,1,8,23,.03,True)
    initial=m.up.detach().clone();fit(m,w,cfg(),23)
    assert not torch.equal(initial,m.up)
    p=tmp_path/'state.npz';save_model(m,p)
    re=load_model(p)
    with torch.no_grad():assert torch.equal(m(w['x_val']),re(w['x_val']))
    assert metrics(m,w,'val')==metrics(re,w,'val')
    p2=tmp_path/'state2.npz';save_model(re,p2)
    assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(p2.read_bytes()).digest()

def test_fixed_learned_share_initial_model():
    a=Model('fixed',32,1,8,29,.03,True);b=Model('learned',32,1,8,29,.03,True)
    x=torch.randn(12,64)
    assert torch.equal(a(x),b(x))
    assert torch.equal(a.up,b.up)
    assert not a.a.requires_grad and b.a.requires_grad

def test_all_joint_gradients_nonzero():
    w=make_world(29,'digits8');m=Model('learned',32,1,8,29,.03,True)
    torch.nn.functional.binary_cross_entropy_with_logits(m(w['x_train'][:32]),w['y_train'][:32]).backward()
    for p in (m.up,m.down,m.head,m.a,m.b,m.code,*m.trunk.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0

def test_one_shared_trunk_call():
    m=Model('learned',32,1,8,29,.03,True);calls=[]
    h=m.trunk.register_forward_hook(lambda *args:calls.append(1))
    assert m(torch.randn(12,64)).shape==(12,8,1)
    h.remove();assert len(calls)==1

def test_task_selection_matches_full_forward():
    m=Model('learned',24,6,12,29,.03,False);x=torch.randn(10,24);tid=torch.arange(10)%12
    assert torch.allclose(m(x,tid),m(x)[torch.arange(10),tid],atol=2e-6)

def test_seed_reproducibility():
    a=make_world(26,'aligned');b=make_world(26,'aligned')
    assert torch.equal(a['x_train'],b['x_train']) and torch.equal(a['y_test'],b['y_test'])
