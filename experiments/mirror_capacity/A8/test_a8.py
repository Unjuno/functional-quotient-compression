# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from a8 import K, KINDS, make_world, create, serialize, deserialize, exact_construction, rank_certificate, train_one, quality


def test_world_generation_is_deterministic_and_split_specific():
    for family in ('hinge', 'ramp'):
        a=make_world(809,family); b=make_world(809,family)
        assert a['fingerprint']==b['fingerprint']
        assert np.array_equal(a['thresholds'], b['thresholds'])
        assert a['train'][0].shape == (1024,1)
        assert a['dev'][0].shape == (384,1)
        assert a['iid'][1].shape == (384,K)
        assert a['ood'][1].shape == (384,K)
        assert not torch.equal(a['dev'][0],a['iid'][0])
        assert not torch.equal(a['iid'][0],a['ood'][0])
        assert (np.diff(a['thresholds']) > 0).all()
        if family=='ramp': assert (a['right'] > a['thresholds']).all()


def test_serializer_roundtrip_bytes_and_cap():
    for cap in (256,512):
        for kind in KINDS:
            m=create(kind,cap)
            raw=serialize(m)
            assert len(raw)<=cap
            assert len(raw)==64+4*sum(p.numel() for p in m.parameters())
            n=deserialize(raw)
            assert serialize(n)==raw
            x=torch.randn(12,1); task=torch.arange(12,dtype=torch.long)
            assert torch.equal(m(x,task),n(x,task))
            with pytest.raises(ValueError): deserialize(raw[:-1]+b'X')


def test_analytic_mirror_construction_is_exact_and_nonunique():
    for family in ('hinge','ramp'):
        w=make_world(809,family)
        for kind in ('mirror','pre'):
            m=exact_construction(kind,w)
            x,y=w['iid']
            ids=torch.arange(K).repeat_interleave(len(x))
            pred=m(x.repeat(K,1),ids).reshape(K,-1).T
            assert torch.allclose(pred,y,atol=2e-6,rtol=2e-6)


def test_rank_certificate_is_a_proven_lower_bound():
    w=make_world(809,'hinge')
    cert=rank_certificate(w, width=3)
    assert cert['target_rank_centered']==K
    assert cert['width']==3
    assert cert['mse_floor']>0
    # The optimal rank-h reconstruction attains the bound when H is unconstrained.
    y=cert['centered_matrix']
    u,s,vt=np.linalg.svd(y,full_matrices=False)
    fitted=(u[:,:3]*s[:3])@vt[:3,:]
    mse=np.mean((y-fitted)**2)
    assert np.isclose(mse,cert['mse_floor'],rtol=1e-9)


def test_bad_inputs_and_training_reproducibility():
    with pytest.raises(ValueError):make_world(-1,'hinge')
    with pytest.raises(ValueError):make_world(809,'invalid')
    with pytest.raises(ValueError):create('invalid',256)
    w=make_world(809,'hinge')
    a=train_one(w,'mirror',256,0,steps=7)
    b=train_one(w,'mirror',256,0,steps=7)
    assert serialize(a['model'])==serialize(b['model'])
    assert np.isfinite(a['train_final_mse'])
    result=quality(a['model'],w['dev'])
    assert result['mse']>=0 and np.isfinite(result['mse'])


def test_validation_selection_cannot_choose_using_iid():
    from audit import summarize
    import pandas as pd
    rows=[]
    for kind,val,iid in [('mirror',.01,.001),('shift',.20,.001),('multihead',.04,.50),
                         ('pre',.50,.005),('film',.08,.05),('basis',.50,.4),('input_affine',.7,.6)]:
        for seed in (0,1):
            for split in ('dev','iid','ood'):
                rows.append(dict(world=810,family='hinge',seed=seed,cap=256,kind=kind,
                                 split=split,mse=val if split=='dev' else iid,nrmse=.4,
                                 max_abs=1.,target_var=1.,bytes=200))
    contrast, frontier = summarize(pd.DataFrame(rows))
    assert len(contrast)==1
    assert contrast.iloc[0].selected_control == 'multihead'
    assert contrast.iloc[0].iid_mirror_mse==.001
    assert contrast.iloc[0].iid_control_mse==.50
    assert 'quality_pass' in frontier.columns
