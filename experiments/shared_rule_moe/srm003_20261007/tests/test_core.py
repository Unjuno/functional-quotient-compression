import sys, pathlib, copy
import pytest
import numpy as np
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'source'))
import core

torch.set_num_threads(1)

def cfg(kind='hybrid'):
    return dict(kind=kind,d=32,ff=48,final_ff=48,heads=4,layers=2,vocab=43,length=5,
                atoms=8,arank=2,experts=8,rank=4,expert_ff=16,topk=2)

def test_world_pair_split_and_atomic_coverage():
    w=core.make_world(66000)
    groups=[set(map(tuple,w[s+'_pairs'].tolist())) for s in ('train','dev','audit')]
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    assert len(set.union(*groups))==24*23
    for group in groups:
        assert all((b,a) in group for a,b in group)
    assert w['atomic_tokens'].shape==(24*16,5)
    assert torch.equal(w['table'].sort(1).values,torch.arange(16).repeat(24,1))
    a,b=w['audit_pairs'].T
    outputs=w['table'][b[:,None],w['table'][a]]
    reverse=w['table'][a[:,None],w['table'][b]]
    assert (outputs!=reverse).float().mean()>.5

def test_data_target_from_tokens_only():
    w=core.make_world(66000)
    for split in ('train','dev','audit'):
        t,y=w[split+'_tokens'],w[split+'_targets']
        q1,q2=t[:,2]-19,t[:,3]-19
        expected=3+w['table'][q2,w['table'][q1,t[:,1]-3]]
        assert torch.equal(y,expected)
    assert not (w['atomic_tokens'][:,2]==0).any().item()

@pytest.mark.parametrize('kind',['dense','moe','lora','shared','hybrid'])
def test_forward_grad_and_serialization(kind):
    c=cfg(kind);m=core.make_model(c,123)
    w=core.make_world(66000); x=w['train_tokens'][:8]
    y=m(x); assert y.shape==(8,43)
    y.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    data=core.encode_model(m); n=core.decode_model(data)
    assert data==core.encode_model(n)
    assert torch.equal(m(x),n(x))
    assert 'world_seed' not in n.config and 'table' not in n.state_dict()

def test_causality_prefix_invariance():
    m=core.make_model(cfg(),1).eval()
    w=core.make_world(66000); x=w['train_tokens'][:8].clone(); z=x.clone();z[:,3:]=torch.randint(3,43,(8,2))
    assert torch.equal(m(x,return_all=True)[:,:3],m(z,return_all=True)[:,:3])

def test_compaction_really_removes_and_preserves_zero_branch():
    m=core.make_model(cfg(),2).eval();n=core.compact(m,[0,2,5,6])
    w=core.make_world(66000);x=w['audit_tokens'][:16]
    assert len(core.encode_model(n))<len(core.encode_model(m))
    assert torch.equal(m(x),n(x)) # zero-up initialization means all residual outputs are zero
    assert n.residual.private.down.shape[0]==4
    assert n.residual.private.router.weight.shape[0]==4
    assert n.residual.private.slot_ids.tolist()==[0,2,5,6]

def test_compaction_empty_and_invalid():
    m=core.make_model(cfg(),2)
    empty=core.compact(m,[])
    x=core.make_world(66000)['audit_tokens'][:4]
    assert torch.equal(m(x),empty(x))
    with pytest.raises(ValueError):core.compact(m,[0,0])
    with pytest.raises(ValueError):core.compact(m,[9])

def test_identical_parent_and_function_preserving_insert():
    parent=core.make_model(cfg('dense'),50)
    x=core.make_world(66000)['train_tokens'][:10]
    for kind in ('moe','lora','shared','hybrid'):
        m=core.make_model(cfg(kind),61,parent)
        assert torch.equal(parent(x),m(x))
