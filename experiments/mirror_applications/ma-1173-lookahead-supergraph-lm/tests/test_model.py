import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'source'))
import torch
from model import *
from data import corpus

def test_full_graph_and_single_tensor_core():
    m=TinyCausalLM('mirror')
    assert EDGES==18 and m.graph.edge_weights().shape==(18,2,12,12)
    assert sum(p.numel() for p in m.parameters())==sum(p.numel() for p in TinyCausalLM('affine').parameters())
    assert len(list(m.graph.parameters()))==6

def test_prune_reduces_real_active_edge_decoding():
    m=TinyCausalLM('mirror')
    x=corpus('balanced','dev',11,2)[:,:-1]
    full=m(x)
    n0=m.deploy_bytes()
    m.graph.cut(0)
    assert m.graph.active_count()==17
    assert m.graph.edge_weights(torch.nonzero(m.graph.active).flatten()).shape[0]==17
    assert m.graph.deployed_edge_state()['addresses'].shape[0]==17
    assert m.deploy_bytes()<=n0  # zip alignment can hide the first row
    for i in (1,2,3,4):m.graph.cut(i)
    assert m.deploy_bytes()<n0
    assert not torch.allclose(m(x),full)

def test_causality_no_future_leak():
    m=TinyCausalLM('mirror')
    seq=corpus('balanced','dev',12,2)
    a=m(seq)
    changed=seq.clone();changed[:,8:]=5
    b=m(changed)
    assert torch.allclose(a[:,:8],b[:,:8],atol=2e-6,rtol=2e-6)

def test_gradient_shared_core_address_gates():
    m=TinyCausalLM('mirror')
    data=corpus('shortcut','train',11,8)
    loss=m.model_loss(data[:,:-1],data[:,1:]);loss.backward()
    assert m.graph.core.grad.abs().sum()>0
    assert m.graph.address.grad.abs().sum()>0
    assert m.graph.logits.grad.abs().sum()>0

def test_mirror_and_affine_non_identical():
    x=corpus('balanced','dev',11,4)[:,:-1]
    a=initialize_model(33,'affine');m=initialize_model(33,'mirror')
    assert a.graph.core.shape==m.graph.core.shape
    assert not torch.allclose(a.graph.edge_weights(),m.graph.edge_weights())
    assert m(x).shape==(4,11,24)

def test_no_oracle_labels_in_training_data():
    seq=corpus('balanced','train',11,100)
    assert torch.all(seq[:,10].ge(5)&seq[:,10].le(10))
    assert torch.all(seq[:,11].ge(11)&seq[:,11].le(16))
    assert not torch.equal(corpus('balanced','train',11,100),corpus('balanced','fresh',11,100))
