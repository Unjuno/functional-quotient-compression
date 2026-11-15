"""A11 family, packing, training and audit guards. Tests written before A11 implementation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pytest
import torch
from a11 import FAMILIES, METHODS, CAP_OPTIONS, make_world, make_model, serialize, deserialize, fit_two_stage, count_bytes


def test_worlds_deterministic_distinct_families_and_disjoint_inputs():
    assert set(FAMILIES)=={'ramp','tent','sine'}
    traces=[]
    for family in FAMILIES:
        w=make_world(1100,family)
        assert w['fingerprint']==make_world(1100,family)['fingerprint']
        assert w['train_y'].shape==(1024,16)
        assert w['support_y'].shape==(8,32)
        assert w['new_iid_y'].shape==(256,8)
        assert w['new_ood_y'].shape==(256,8)
        assert torch.isfinite(w['train_y']).all()
        assert len(set(w['meta_x'].flatten().tolist())&set(w['new_iid_x'].flatten().tolist()))==0
        traces.append(w['fingerprint'])
    assert len(set(traces))==3


def test_stored_bytes_and_roundtrip_across_caps():
    for method in METHODS:
        for cap in CAP_OPTIONS:
            model=make_model(method,cap,11)
            b=serialize(model)
            assert len(b)==count_bytes(model)<=cap
            assert serialize(deserialize(b))==b


def test_broken_digest_is_rejected():
    b=bytearray(serialize(make_model('mirror',512,3)))
    b[41]^=1
    with pytest.raises(ValueError,match='digest'):
        deserialize(bytes(b))


def test_no_audit_labels_used_in_fit():
    w=make_world(1101,'sine')
    wrong=dict(w)
    wrong['meta_dev_y']=torch.full_like(w['meta_dev_y'],float('nan'))
    wrong['new_iid_y']=torch.full_like(w['new_iid_y'],float('nan'))
    wrong['new_ood_y']=torch.full_like(w['new_ood_y'],float('nan'))
    a=fit_two_stage(w,'shift',512,0,meta_steps=3,adapt_steps=2,batch=16)
    b=fit_two_stage(wrong,'shift',512,0,meta_steps=3,adapt_steps=2,batch=16)
    assert serialize(a['model'])==serialize(b['model'])


def test_two_stage_adaptation_only_changes_new_codes():
    model=make_model('film',640,17)
    model.enter_adaptation()
    assert [n for n,p in model.named_parameters() if p.requires_grad]==['new_codes']


def test_meta_training_smoke_lock_then_independent_audit(tmp_path):
    from train import run_training
    from audit import audit_run
    config=dict(experiment='A11_SMOKE',worlds=[1100],families=['tent'],seeds=[0],methods=['mirror','shift'],
                caps=[512],meta_steps=3,adapt_steps=2,batch=16,workers=1)
    out=tmp_path/'test'
    r=run_training(out,config)
    assert len(r['models'])==2 and not (out/'audit.csv').exists()
    s=audit_run(out)
    assert s['n_models']==2 and s['n_scores']==6


def test_corrupted_weight_fails_before_score(tmp_path):
    from train import run_training
    from audit import audit_run
    config=dict(experiment='A11_SMOKE',worlds=[1100],families=['ramp'],seeds=[0],methods=['mirror'],
                caps=[512],meta_steps=2,adapt_steps=2,batch=16,workers=1)
    out=tmp_path/'test';r=run_training(out,config)
    p=out/r['models'][0]['path'];data=bytearray(p.read_bytes());data[50]^=1;p.write_bytes(data)
    with pytest.raises(RuntimeError,match='hash'):
        audit_run(out)


def test_protocol_change_fails_before_score(tmp_path,monkeypatch):
    from train import run_training
    from audit import audit_run
    config=dict(experiment='A11_SMOKE',worlds=[1100],families=['ramp'],seeds=[0],methods=['shift'],
                caps=[512],meta_steps=2,adapt_steps=2,batch=16,workers=1)
    out=tmp_path/'test';r=run_training(out,config)
    z=json.loads((out/'train.lock.json').read_text());z['protocol_sha256']='0'*64
    (out/'train.lock.json').write_text(json.dumps(z))
    with pytest.raises(RuntimeError,match='protocol'):
        audit_run(out)


def test_replay_selection_and_byte_identical_regeneration(tmp_path):
    from train import run_training
    from replay import select_replay_cases, verify_replay
    cfg=dict(experiment='A11_REPLAY_TEST',worlds=[1101],families=['sine'],seeds=[1],
             methods=['film','mirror'],caps=[512],meta_steps=3,adapt_steps=2,batch=16,workers=1)
    out=tmp_path/'locked';r=run_training(out,cfg,workers=1)
    cases=select_replay_cases(r['models'],2)
    assert len(cases)==2 and {c['method'] for c in cases}=={'film','mirror'}
    rec=verify_replay(out,limit=2)
    assert rec['passed']==2 and rec['failed']==0


def test_observed_frontier_censors_failing_families_and_uses_actual_bytes():
    from analyze import observed_frontier
    rows=[
      dict(family='ramp',world=1,method='mirror',cap=768,stored_bytes=764,passes=1),
      dict(family='ramp',world=1,method='shift',cap=512,stored_bytes=500,passes=1),
      dict(family='ramp',world=1,method='film',cap=640,stored_bytes=624,passes=1),
      dict(family='sine',world=1,method='mirror',cap=512,stored_bytes=500,passes=0),
      dict(family='sine',world=1,method='shift',cap=512,stored_bytes=500,passes=0),
    ]
    out=observed_frontier(rows)
    r=next(x for x in out if x['family']=='ramp')
    s=next(x for x in out if x['family']=='sine')
    assert r['mirror_bytes']==764 and r['reference_bytes']==500
    assert abs(r['observed_ratio']-1.528)<1e-12 and r['mirror_10pct_win'] is False
    assert s['observed_ratio'] is None and s['classification']=='both_censored'
