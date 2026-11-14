import hashlib
from pathlib import Path
import numpy as np
import pytest
import torch

from a10 import (METHODS, CAP_OPTIONS, K_TRAIN, K_NEW, make_world,
                 Student, make_model, serialize, deserialize, count_bytes,
                 fit_two_stage, metric)


def test_worlds_are_deterministic_and_evaluation_inputs_disjoint():
    a, b = make_world(900), make_world(900)
    assert a['fingerprint'] == b['fingerprint']
    assert a['support_x'].shape == (K_NEW, 32, 1)
    assert a['train_y'].shape == (1024, K_TRAIN)
    assert a['new_iid_y'].shape == (512, K_NEW)
    train=set(a['meta_x'][:, 0].numpy().tobytes()[i:i+4] for i in range(0,len(a['meta_x'])*4,4))
    iid=set(a['new_iid_x'][:, 0].numpy().tobytes()[i:i+4] for i in range(0,len(a['new_iid_x'])*4,4))
    assert train.isdisjoint(iid)
    assert not np.allclose(a['new_iid_y'].numpy(),make_world(901)['new_iid_y'].numpy())


def test_serialization_roundtrip_all_methods():
    torch.set_num_threads(1)
    for method in METHODS:
        m=make_model(method,512,seed=22)
        raw=serialize(m)
        assert len(raw)==count_bytes(m) and len(raw)<=512
        n=deserialize(raw)
        x=torch.tensor([[-.2],[.7],[1.8]])
        ids=torch.tensor([0,K_TRAIN,K_TRAIN+K_NEW-1])
        assert torch.equal(m(x,ids),n(x,ids))
        assert serialize(n)==raw
        assert raw[-32:]==hashlib.sha256(raw[:-32]).digest()


def test_flipped_checksum_rejected():
    raw=bytearray(serialize(make_model('mirror',512,seed=4)))
    raw[35]^=1
    with pytest.raises(ValueError,match='digest'):
        deserialize(bytes(raw))


def test_model_caps_and_adapter_parameter_isolation():
    for method in METHODS:
        for cap in CAP_OPTIONS:
            m=make_model(method,cap,seed=5)
            assert len(serialize(m))<=cap
            m.enter_adaptation()
            names=[name for name,p in m.named_parameters() if p.requires_grad]
            assert names==(['head_new.weight','head_new.bias'] if method=='multihead' else ['new_codes'])


def test_zero_shear_reduces_to_shared_relu():
    m=make_model('mirror',512,seed=2)
    with torch.no_grad():
        m.train_codes.zero_(); m.new_codes.zero_()
        x=torch.tensor([[-1.],[-.1],[.7]])
        tasks=torch.tensor([0,1,16])
        shared=m.down(torch.relu(m.up(x))).flatten()
        assert torch.allclose(m(x,tasks),shared,atol=1e-6,rtol=0)


def test_metric_matches_manual_task_centered_normalization():
    target=torch.tensor([[0.,1.],[1.,3.],[2.,5.]])
    pred=target+1
    d=metric(pred,target)
    assert abs(d['mse']-1)<1e-7
    assert abs(d['nrmse']-np.sqrt(1/((2/3+8/3)/2)))<1e-6


def test_freeze_keeps_shared_weights_bit_identical_during_adaptation():
    w=make_world(900)
    m=make_model('shift',512,seed=5)
    first={k:v.detach().clone() for k,v in m.state_dict().items() if k!='new_codes'}
    # Tiny real training cycle, adaptation only; test must not rely on mocks.
    m.enter_adaptation()
    optim=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.02)
    for _ in range(4):
        xx=w['support_x'].reshape(-1,1)
        ii=torch.arange(K_TRAIN,K_TRAIN+K_NEW).repeat_interleave(32)
        optim.zero_grad()
        loss=(m(xx,ii)-w['support_y'].reshape(-1)).square().mean()
        loss.backward(); optim.step()
    for k,old in first.items():
        assert torch.equal(m.state_dict()[k],old),k


def test_training_reports_actual_meta_loss_not_placeholder():
    world=make_world(899)
    r=fit_two_stage(world,'mirror',512,1,meta_steps=2,adapt_steps=2,batch=16)
    assert np.isfinite(r['meta_final_batch_mse'])
    assert r['meta_final_batch_mse']>0


def test_train_lock_and_independent_audit(tmp_path):
    from train import run_training
    from audit import audit_run
    config=dict(worlds=[899],seeds=[0],methods=['mirror'],caps=[512],
                meta_steps=3,adapt_steps=2,batch=16,experiment='A10_SMOKE')
    out=tmp_path/'run'
    done=run_training(out,config,workers=1)
    assert len(done['models'])==1
    assert (out/'train.lock.json').is_file()
    assert not (out/'audit.csv').exists()
    a=audit_run(out)
    assert a['n_models']==1
    assert (out/'audit.csv').is_file()
    assert len(a['worlds'])==1
    assert a['worlds'][0]['world']==899
    assert np.isfinite(a['worlds'][0]['new_iid_nrmse'])


def test_training_does_not_overwrite_existing_lock(tmp_path):
    from train import run_training
    config=dict(worlds=[899],seeds=[0],methods=['pre'],caps=[512],
                meta_steps=2,adapt_steps=2,batch=16,experiment='A10_SMOKE')
    out=tmp_path/'run'
    run_training(out,config,workers=1)
    with pytest.raises(FileExistsError):
        run_training(out,config,workers=1)


def test_replay_selection_spans_available_method_families():
    from replay import choose_cases
    items=[{'method':m,'world':w} for w in range(6) for m in METHODS]
    selected=choose_cases(items,8)
    assert len(selected)==8
    assert len({v['method'] for v in selected})==len(METHODS)


def test_locked_audit_rejects_corrupted_model(tmp_path):
    from train import run_training
    from audit import audit_run
    c=dict(worlds=[899],seeds=[0],methods=['mirror'],caps=[512],meta_steps=2,adapt_steps=2,batch=8,experiment='A10_SMOKE')
    out=tmp_path/'run';r=run_training(out,c,workers=1)
    p=out/r['models'][0]['path']
    b=bytearray(p.read_bytes());b[37]^=1;p.write_bytes(b)
    with pytest.raises(RuntimeError,match='model hash'):
        audit_run(out)


def test_changing_withheld_audit_labels_cannot_change_trained_model():
    world=make_world(899)
    tainted=world.copy()
    tainted['new_iid_y']=torch.full_like(world['new_iid_y'],float('nan'))
    tainted['new_ood_y']=torch.full_like(world['new_ood_y'],float('nan'))
    tainted['meta_dev_y']=torch.full_like(world['meta_dev_y'],float('nan'))
    tainted['teacher_left']=np.full_like(world['teacher_left'],float('nan'))
    tainted['teacher_right']=np.full_like(world['teacher_right'],float('nan'))
    a=fit_two_stage(world,'mirror',512,0,meta_steps=4,adapt_steps=3,batch=16)
    b=fit_two_stage(tainted,'mirror',512,0,meta_steps=4,adapt_steps=3,batch=16)
    assert serialize(a['model'])==serialize(b['model'])
