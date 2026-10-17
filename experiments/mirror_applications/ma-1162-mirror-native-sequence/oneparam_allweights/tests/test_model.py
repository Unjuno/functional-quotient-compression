import io
import math
import pytest
import torch
from source.model import Transformer, conjugate, batch_data, world_encoding, pack, unpack

torch.set_num_threads(1)


def test_only_core_and_scalar_addresses_are_learned():
    model = Transformer('mirror')
    assert {k: tuple(v.shape) for k,v in model.named_parameters()} == {'core': (64,64), 'codes': (14,)}
    assert sum(p.numel() for p in model.parameters()) == 4110
    assert not any(isinstance(m, (torch.nn.Linear,torch.nn.Embedding,torch.nn.LayerNorm)) for m in model.modules())


def test_literal_one_scalar_not_one_parameter_tensor():
    model = Transformer('scalar_only')
    assert sum(p.numel() for p in model.parameters()) == 1
    assert model.core.numel() == 4096 and not model.core.requires_grad
    assert len(pack(model)) > 4096 * 4


def test_conjugation_is_orthogonal_and_changes_matrix():
    torch.manual_seed(51)
    s=torch.randn(64,64,dtype=torch.float64)
    y=conjugate(s,torch.tensor([.123,.456],dtype=torch.float64))
    assert torch.allclose(torch.linalg.svdvals(y[0]),torch.linalg.svdvals(s),atol=1e-10)
    assert torch.allclose(y[0].norm(),s.norm(),atol=1e-10)
    assert (y[0]-y[1]).abs().mean() > .05
    assert torch.allclose(conjugate(y[0],torch.tensor([-.123],dtype=torch.float64))[0],s,atol=1e-10)


@pytest.mark.parametrize('mode',['mirror','shared','gain','dense','dense_byte','scalar_only'])
def test_every_model_is_causal_and_roundtrips(mode):
    torch.manual_seed(52)
    model=Transformer(mode).eval()
    x=torch.randint(0,64,(2,16)); x2=x.clone(); x2[:,9:]=torch.randint(0,64,(2,7))
    with torch.no_grad():
        a=model(x,all_positions=True); b=model(x2,all_positions=True)
        clone=unpack(pack(model))
        c=clone(x,all_positions=True)
    assert a.shape==(2,16,64) and torch.isfinite(a).all()
    assert torch.allclose(a[:,:9],b[:,:9],atol=1e-6,rtol=0)
    assert torch.equal(a,c)


def test_gradients_reach_core_and_every_role_code():
    torch.manual_seed(55)
    m=Transformer('mirror')
    loss=torch.nn.functional.cross_entropy(m(torch.randint(0,64,(8,16))),torch.randint(0,8,(8,)))
    loss.backward()
    assert torch.isfinite(m.core.grad).all() and m.core.grad.norm()>0
    assert torch.isfinite(m.codes.grad).all() and (m.codes.grad.abs()>1e-9).all()


def test_byte_near_dense_parameter_count():
    assert sum(p.numel() for p in Transformer('dense_byte').parameters()) == 4128
    assert abs(len(pack(Transformer('dense_byte')))/len(pack(Transformer('mirror')))-1)<.05


@pytest.mark.parametrize('task',['copy','retrieve'])
def test_task_label_matches_observed_causal_prefix(task):
    perm=world_encoding(618230); inv=perm.argsort()
    x,y=batch_data(task,512,torch.Generator().manual_seed(7),perm)
    raw=inv[x]
    if task=='copy':
        expected=raw.gather(1,(15-(raw[:,-1]-48))[:,None]).squeeze(1)
    else:
        key=raw[:,-1]-32
        expected=((raw[:,:8]%4)*(raw[:,:8]//4==key[:,None])).sum(-1)
        assert (raw[:,:8]//4==key[:,None]).sum(-1).eq(1).all()
    assert torch.equal(y,expected)


def test_equal_data_streams_and_separate_eval():
    p=world_encoding(618230)
    a=batch_data('copy',64,torch.Generator().manual_seed(19),p)
    b=batch_data('copy',64,torch.Generator().manual_seed(19),p)
    c=batch_data('copy',64,torch.Generator().manual_seed(20),p)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert not torch.equal(a[0],c[0])
