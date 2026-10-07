import copy, sys, pathlib
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'source'))
from core import *
from engine import *

def test_selection_cap_and_parent_anchor():
    configs,cap=select_configs();parent=make_model(BASE,1)
    x=make_world(66000)['train_tokens'][:16]
    for c in configs.values():
        m=make_model(c,2,parent)
        assert len(encode_model(m))<=cap
        assert torch.allclose(m(x),parent(x),atol=2e-6,rtol=2e-6)

def test_stream_replay_prefix_and_atomic_coverage():
    w=make_world(66000)
    a,b=stream(w,16,20);c,d=stream(w,16,20)
    assert torch.equal(a,c) and torch.equal(b,d)
    atoms=a[:8,:48].reshape(-1,5)
    assert len(torch.unique(atoms,dim=0))==384
    pair,_=stream(w,16,20,True)
    assert (pair[:,:,4]==2).all()

def test_metrics_exact_and_pruning_not_access_private_truth():
    w=make_world(66000);configs,_=select_configs();m=make_model(configs['hybrid'],1)
    w.pop('private_truth');w.pop('table');w.pop('audit_tokens');w.pop('audit_targets');w.pop('audit_pairs')
    child,log=prune_validation(m,w,4)
    assert child.residual.private.n==4
    assert log['candidate_evaluations']==26
    assert 'dev_acc' in log['history'][-1]

def test_serialization_corruption_rejected():
    m=make_model(BASE,1);raw=encode_model(m)
    import pytest
    with pytest.raises(ValueError):decode_model(raw[:-1])
    with pytest.raises(ValueError):decode_model(raw+b'junk')
    with pytest.raises(ValueError):decode_model(b'bad'+raw[3:])

def test_active_topk_bank_matches_dense_reference():
    torch.manual_seed(4)
    for signed in (False,True):
        bank=SparseBank(8,5,3,2,signed=signed)
        torch.nn.init.normal_(bank.up)
        x=torch.randn(7,8);scores=bank.router(x)
        ids=(scores.abs() if signed else scores).topk(2,-1).indices
        c=scores.gather(1,ids);c=torch.tanh(c)/2 if signed else c.softmax(-1)
        expected=torch.stack([sum(c[b,j]*(x[b]@bank.down[ids[b,j]]@bank.up[ids[b,j]]) for j in range(2)) for b in range(7)])
        assert torch.allclose(bank(x),expected,atol=2e-6)
