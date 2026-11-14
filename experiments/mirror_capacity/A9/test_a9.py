import numpy as np
import pytest
import torch

from experiments.mirror_capacity.A8 import a8
from experiments.mirror_capacity.A9 import a9


def test_exact_scaffolds_equal_and_teacher_independent():
    xx=torch.linspace(-2.0,2.0,64).reshape(-1,1).repeat(16,1)
    ids=torch.arange(16).repeat_interleave(64)
    mm={k:a9.initialize(k,'scaffold_code4',820,0) for k in ('mirror','pre','shift')}
    with torch.no_grad():
        ys=[mm[k](xx,ids) for k in mm]
    assert (ys[0]-ys[1]).abs().max().item() < 1e-7
    assert (ys[0]-ys[2]).abs().max().item() < 1e-7
    assert torch.equal(mm['mirror'].code.weight,mm['shift'].code.weight)
    assert not torch.allclose(mm['mirror'].code.weight[:,0],torch.tensor(a8.make_world(820,'ramp')['thresholds']))


def test_code_gradients_match_on_scaffold():
    xx=torch.linspace(-1.8,1.8,64).reshape(-1,1).repeat(16,1)
    ids=torch.arange(16).repeat_interleave(64)
    gradients=[]
    for kind in ('mirror','pre','shift'):
        model=a9.initialize(kind,'scaffold_code4',820,1)
        out=model(xx,ids)
        out.square().mean().backward()
        gradients.append(model.code.weight.grad.detach().clone())
    assert max((g-gradients[0]).abs().max().item() for g in gradients[1:]) < 1e-6


def test_trainable_parameter_freeze_and_byte_count():
    for kind in ('mirror','pre','shift'):
        m=a9.initialize(kind,'scaffold_code4',820,0)
        assert [name for name,p in m.named_parameters() if p.requires_grad]==['code.weight']
        expected=244 if kind in ('mirror','pre') else 276
        assert len(a8.serialize(m))==expected
        reloaded=a8.deserialize(a8.serialize(m))
        assert a8.serialize(reloaded)==a8.serialize(m)
    wide=a9.initialize('mirror','random_wide',820,0)
    assert len(a8.serialize(wide))<=512 and wide.width>4


def test_noise_changes_shared_weights_but_not_code():
    a=a9.initialize('mirror','scaffold_full4',821,0)
    b=a9.initialize('mirror','scaffold_noisy4',821,0)
    assert torch.equal(a.code.weight,b.code.weight)
    assert not torch.equal(a.up.weight,b.up.weight)
    assert torch.equal(b.up.weight,a9.initialize('mirror','scaffold_noisy4',821,0).up.weight)


def test_random_code_only_freezes_shared_weights():
    m=a9.initialize('mirror','random_code4',820,0)
    assert [name for name,p in m.named_parameters() if p.requires_grad]==['code.weight']
    assert torch.isfinite(m.code.weight).all()


def test_training_deterministic_and_can_be_audited():
    data=a8.make_world(819,'ramp')
    a=a9.fit(data,'mirror','scaffold_code4',0,steps=12)
    b=a9.fit(data,'mirror','scaffold_code4',0,steps=12)
    assert a8.serialize(a['model'])==a8.serialize(b['model'])
    assert np.isfinite(a8.quality(a['model'],data['iid'])['nrmse'])


def test_invalid_regime_fails():
    with pytest.raises(ValueError):
        a9.initialize('mirror','unknown',820,0)
