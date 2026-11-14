import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pytest
import torch
from a7 import make_world, task_templates, source_core


def test_template_shape_centering_norm():
    templates=task_templates()
    assert templates.shape == (8,64)
    assert np.allclose(templates.sum(axis=1), 0., atol=2e-6)
    assert np.allclose(np.linalg.norm(templates,axis=1), 1., atol=1e-6)


def test_pairs_labels_split_and_determinism():
    for family in ('pair_and','pair_xor'):
        a=make_world(701,family); b=make_world(701,family)
        ids=a['split_ids']
        assert set(ids['train']).isdisjoint(ids['val'])
        assert set(ids['train']).isdisjoint(ids['iid'])
        assert set(ids['val']).isdisjoint(ids['iid'])
        assert a['pairs'].shape==(16,2)
        assert len({tuple(p) for p in a['pairs']})==16
        assert np.array_equal(a['pairs'],b['pairs'])
        assert np.array_equal(a['thresholds'],b['thresholds'])
        assert torch.equal(a['iid'][1],a['ood'][1])
        assert not torch.equal(a['iid'][0],a['ood'][0])
        assert torch.equal(a['iid'][1],b['iid'][1])
        assert a['train'][0].shape[1]==64
        p=a['train'][1].float().mean(dim=1).numpy()
        if family=='pair_and': assert np.all((p>=0.18)&(p<=0.40))
        else: assert np.all((p>=0.30)&(p<=0.70))


def test_labels_match_independently_recomputed_from_raw_pixels():
    a=make_world(700,'pair_xor')
    img=a['iid'][0].numpy()
    scores=img@task_templates().T
    bits=scores>a['thresholds'][None,:]
    fresh=np.stack([bits[:,i]^bits[:,j] for i,j in a['pairs']]).astype(np.float32)
    assert np.array_equal(fresh,a['iid'][1].numpy())


def test_source_model_repro_and_exact_fp32_bytes():
    core=source_core()
    for kind in core.KINDS:
        m=core.create(kind,6000); raw=core.serialize(m)
        assert len(raw)<=6000
        y=core.deserialize(raw)
        assert core.serialize(y)==raw
        x=torch.rand(3,64); t=torch.arange(3)
        assert torch.equal(m(x,t),y(x,t))


def test_audit_control_selected_by_validation_not_test():
    from A7.audit import summarize
    import pandas as pd
    rows=[]
    for kind, val_bce, iid_bce in [('mirror4',0.2,0.1),('film4',0.3,0.01),('multihead',0.1,0.5)]:
        for split in ('val','iid','ood'):
            rows.append({'world':700,'family':'pair_and','cap':6000,'seed':0,'kind':kind,'split':split,
                        'bce':val_bce if split=='val' else iid_bce,'balanced_accuracy':0.8,
                        'accuracy':0.8,'worst_task_bce':1.,'bytes':5800})
    contrasts,_,_,_=summarize(pd.DataFrame(rows),allow_single_world=True)
    assert contrasts.iloc[0].selected_control=='multihead'
    assert contrasts.iloc[0].iid_mirror_bce==0.1
    assert contrasts.iloc[0].iid_control_bce==0.5


def test_wrong_family_rejected():
    with pytest.raises(ValueError):make_world(700,'unrecognized')
    with pytest.raises(ValueError):make_world(-1,'pair_and')
