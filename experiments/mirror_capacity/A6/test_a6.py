import hashlib
import numpy as np
import pytest
import torch
from a6 import KINDS, create, serialize, deserialize, make_world, eval_model, explicit_shear, model_bytes


def test_data_is_disjoint_reproducible():
    a=make_world(600,'mask'); b=make_world(600,'mask')
    assert set(a['split_ids']['train']).isdisjoint(a['split_ids']['iid'])
    assert set(a['split_ids']['val']).isdisjoint(a['split_ids']['iid'])
    assert len(set(map(tuple,a['masks'])))==16
    assert np.array_equal(a['masks'],b['masks'])
    for p in ('train','val','iid','ood'):
        assert torch.equal(a[p][0],b[p][0]); assert torch.equal(a[p][1],b[p][1])
    assert a['iid'][0].shape[1]==64


def test_no_ood_label_change():
    a=make_world(601,'xorink')
    assert torch.equal(a['iid'][1],a['ood'][1])
    assert not torch.equal(a['iid'][0],a['ood'][0])
    assert a['threshold']>0


def test_model_byte_caps_roundtrip():
    for kind in KINDS:
        model=create(kind,6000); raw=serialize(model)
        assert len(raw)==model_bytes(model)<=6000
        out=deserialize(raw)
        assert out.kind==kind
        assert serialize(out)==raw
        x=torch.rand(8,64); t=torch.arange(8)
        assert torch.allclose(model(x,t),out(x,t),rtol=0,atol=0)


def test_payload_tamper_rejected():
    raw=bytearray(serialize(create('mirror4',6000)));raw[47]^=1
    with pytest.raises(ValueError): deserialize(bytes(raw))


def test_shear_explicit_matches():
    x=torch.randn(6,64); t=torch.arange(6)
    for kind in ('mirror4','pre4'):
        model=create(kind,6000)
        assert torch.allclose(model(x,t),explicit_shear(model,x,t),atol=2e-5,rtol=2e-5)


def test_all_codes_affect_output():
    for kind in KINDS:
        model=create(kind,12000)
        x=torch.randn(12,64); y=model(x,torch.zeros(12,dtype=torch.long)); z=model(x,torch.ones(12,dtype=torch.long))
        assert (y-z).abs().max()>1e-10


def test_eval_finite_and_correct_task_shape():
    world=make_world(602,'mask'); model=create('basis4',6000)
    ev=eval_model(model,world['iid'])
    assert np.isfinite(ev['bce'])
    assert 0<=ev['balanced_accuracy']<=1
    assert 0<=ev['accuracy']<=1


def test_invalid_spec_rejected():
    with pytest.raises(ValueError):create('mirror4',100)
    with pytest.raises(ValueError):make_world(-1,'mask')
    with pytest.raises(ValueError):make_world(600,'bogus')
