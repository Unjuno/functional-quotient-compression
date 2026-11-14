import math
import numpy as np
import pytest
import torch
from a5 import KINDS, create, serialize, deserialize, make_world, eval_model, shear_explicit


def test_serializer_roundtrip_and_exact_bytes():
    torch.manual_seed(1)
    for cap in (1500, 3000):
        for kind in KINDS:
            m = create(kind, cap)
            b = serialize(m)
            assert len(b) <= cap
            assert len(b) == 64 + 4 * sum(p.numel() for p in m.parameters())
            m2 = deserialize(b)
            assert serialize(m2) == b
            x = torch.randn(9, 6); task = torch.arange(9) % 16
            assert torch.equal(m(x, task), m2(x, task))


def test_bad_checksum_rejected():
    b = bytearray(serialize(create('mirror4', 1500)))
    b[50] ^= 1
    with pytest.raises(ValueError): deserialize(bytes(b))


def test_shear_explicit_generated_weights():
    torch.manual_seed(2)
    m = create('mirror4', 1500)
    x = torch.randn(7, 6)
    t = torch.arange(7, dtype=torch.long)
    assert torch.allclose(m(x, t), shear_explicit(m, x, t), atol=5e-6, rtol=1e-5)


def test_pre_shear_explicit_generated_weights():
    torch.manual_seed(2)
    m = create('pre4', 1500)
    x = torch.randn(7, 6)
    t = torch.arange(7, dtype=torch.long)
    assert torch.allclose(m(x, t), shear_explicit(m, x, t), atol=5e-6, rtol=1e-5)


def test_task_id_is_effective():
    torch.manual_seed(5)
    m=create('film4', 1500)
    x=torch.randn(6,6)
    assert not torch.allclose(m(x,torch.zeros(6,dtype=torch.long)),m(x,torch.ones(6,dtype=torch.long)))


def test_dataset_determinism_separation():
    for family in ('independent','compositional'):
        d1=make_world(500,family)
        d2=make_world(500,family)
        assert all(torch.equal(d1[k][0],d2[k][0]) and torch.equal(d1[k][1],d2[k][1]) for k in d1)
        assert not torch.equal(d1['train'][0],d1['iid'][0])
        assert torch.var(d1['iid'][1],dim=0).mean().item()>1e-5


def test_eval_finite():
    model=create('basis4',1500)
    data=make_world(500,'compositional')
    got=eval_model(model,data['val'])
    assert got['nrmse']>0 and math.isfinite(got['nrmse'])
    assert math.isfinite(got['mse'])
