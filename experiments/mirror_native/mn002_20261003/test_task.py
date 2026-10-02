# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from task import make_table, dataset, check_splits
from model import TinyLM

def test_each_rule_is_permutation_and_reproducible():
    t = make_table()
    assert t.shape == (16,32) and t.dtype == torch.long
    assert torch.equal(t, make_table())
    assert torch.equal(t.sort(-1).values, torch.arange(32).expand(16,32))

def test_causal_targets_exact_no_answer_at_query():
    t=make_table(); x,y=dataset(16,12001,t)
    assert x.shape == y.shape == (16,12)
    assert (x[:,0] == 48).all()
    assert (y[:,[0,1,3,4,6,7,9,10]] == -100).all()
    for j in range(4):
        rule = x[:,1+3*j]-32; symbol=x[:,2+3*j]
        assert torch.equal(y[:,2+3*j],t[rule,symbol])
        if j<3: assert torch.equal(x[:,3+3*j],y[:,2+3*j])
    assert torch.equal(x, dataset(16,12001,t)[0])

def test_disjoint_sequences_and_all_pairs_seen_in_train():
    t=make_table(); tr=dataset(8192,12001,t); dv=dataset(512,23002,t)
    assert check_splits(tr,dv)['overlap_sequences']==0
    pairs=(tr[0][:,1::3]-32)*32+tr[0][:,2::3]
    assert pairs.unique().numel()==512
    with pytest.raises(ValueError): check_splits(tr,tr)

@pytest.mark.parametrize('states',[1,4,16])
def test_exact_parameter_dense_controls(states):
    mirror=TinyLM(kind='mirror',hidden=32,states=states,vocab=49,max_length=32)
    dense=TinyLM(kind='dense',hidden=32+states,vocab=49,max_length=32)
    assert sum(p.numel() for p in mirror.parameters())==sum(p.numel() for p in dense.parameters())

@pytest.mark.parametrize('n,records',[(0,4),(-1,4),(1,0),(1,-1)])
def test_invalid_sizes_rejected(n,records):
    with pytest.raises(ValueError): dataset(n,0,torch.arange(8).view(2,4),records)

def test_one_state_is_input_independent():
    m=TinyLM(kind='mirror',hidden=32,states=1,vocab=49,max_length=32).eval()
    x,_=dataset(8,5,make_table())
    with torch.no_grad():
        native=m(x)[0]; fixed=m(x,override='uniform')[0]
    assert torch.equal(native,fixed)
