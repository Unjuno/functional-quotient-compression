"""A12 preregistered guard tests. Written before model/training implementation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from a12 import make_support, new_code_start, adapt_branch, fit_meta_checkpoints, audit_run
from a11 import make_world, make_model, serialize, deserialize


def test_nested_support_same_as_a11_exact_and_no_oracle_parameters_exposed():
    for family in ('tent','sine'):
        w=make_world(1200,family)
        x32,y32=make_support(1200,family,32)
        x256,y256=make_support(1200,family,256)
        assert torch.equal(x32,w['support_x'])
        assert torch.equal(y32,w['support_y'])
        assert torch.equal(x256[:,:32],x32)
        assert torch.equal(y256[:,:32],y32)
        assert x256.shape==(8,256,1) and y256.shape==(8,256)
        assert torch.isfinite(y256).all()


def test_initial_new_codes_and_saved_budget():
    for method in ('mirror','shift','film'):
        m=make_model(method,1024,1200*1009+0*71+{'mirror':19,'shift':19+2*1409,'film':19+3*1409}[method])
        initial=new_code_start(1200,method,0,1024)
        assert torch.equal(initial,m.new_codes.detach())
        assert len(serialize(m))<=1024


def test_adaptation_preserves_shared_weights_and_does_not_read_audit_labels():
    w=make_world(1201,'tent')
    model=make_model('shift',1024,123)
    before={name:p.detach().clone() for name,p in model.named_parameters() if name!='new_codes'}
    bogus=dict(w)
    for field in ('meta_dev_y','new_iid_y','new_ood_y'):
        bogus[field]=torch.full_like(w[field],float('nan'))
    _,out1=adapt_branch(model,w,1201,'tent',0,'shift',support_n=32,steps=(3,),batch=16)
    _,out2=adapt_branch(model,bogus,1201,'tent',0,'shift',support_n=32,steps=(3,),batch=16)
    assert serialize(out1[3])==serialize(out2[3])
    assert all(torch.equal(before[k],dict(out1[3].named_parameters())[k]) for k in before)


def test_uninterrupted_meta_has_both_checkpoints_and_no_test_labels():
    w=make_world(1201,'sine')
    a=fit_meta_checkpoints(w,'film',1024,seed=0,steps=(2,5),batch=16)
    wrong=dict(w)
    wrong['meta_dev_y']=torch.full_like(w['meta_dev_y'],float('nan'))
    wrong['new_iid_y']=torch.full_like(w['new_iid_y'],float('nan'))
    wrong['new_ood_y']=torch.full_like(w['new_ood_y'],float('nan'))
    b=fit_meta_checkpoints(wrong,'film',1024,seed=0,steps=(2,5),batch=16)
    assert set(a)=={2,5}
    for k in a:
        assert serialize(a[k])==serialize(b[k])


def test_lock_before_audit_and_tampering_rejected(tmp_path):
    from run import run_experiment
    cfg=dict(experiment='A12_SMOKE',families=['tent'],worlds=[1200],seeds=[0],methods=['mirror'],cap=1024,
             meta_checkpoints=[2,4],support_counts=[32,256],adapt_checkpoints=[2,4],batch=16,workers=1)
    path=tmp_path/'tiny'
    lock=run_experiment(path,cfg)
    assert len(lock['models'])==8
    assert not (path/'audit.csv').exists()
    result=audit_run(path)
    assert result['models']==8 and result['split_scores']==24
    data=json.loads((path/'train.lock.json').read_text())
    data['protocol_sha256']='0'*64
    (path/'train.lock.json').write_text(json.dumps(data))
    with pytest.raises(RuntimeError,match='protocol'):
        audit_run(path,recompute=True)


def test_fp32_deserialization_corruption_rejected():
    raw=bytearray(serialize(make_model('mirror',1024,23)))
    raw[48]^=1
    with pytest.raises(ValueError,match='digest'):
        deserialize(bytes(raw))


def test_family_informed_support_fit_recovers_noise_free_tent_and_sine():
    from oracle import fit_family_from_support, predict_family
    x=np.linspace(-1.5,1.5,51)
    for family,true in [('tent',(.13,.33)),('sine',(1.71,.63,-.37))]:
        if family=='tent':
            center,width=true
            y=np.maximum(0.,1.-np.abs(x-center)/width)
        else:
            amp,freq,phase=true
            y=amp*np.sin(freq*x+phase)
        estimated=fit_family_from_support(family,x,y)
        pred=predict_family(family,x,estimated)
        assert np.mean((pred-y)**2)<1e-8, (family,estimated)


def test_joint_training_reads_no_audit_labels_and_unfreezes_weights():
    from a12b import train_joint_model
    w=make_world(1202,'sine')
    m=make_model('shift',1024,123)
    bogus=dict(w)
    for name in ('meta_dev_y','new_iid_y','new_ood_y'):
        bogus[name]=torch.full_like(w[name],float('nan'))
    a=train_joint_model(m,w,1202,'sine',0,steps=3,batch=16)
    b=train_joint_model(m,bogus,1202,'sine',0,steps=3,batch=16)
    assert serialize(a)==serialize(b)


def test_a12b_meta_parent_has_untouched_new_code_values():
    w=make_world(1201,'tent')
    parent=fit_meta_checkpoints(w,'mirror',1024,0,steps=(3,),batch=16)[3]
    assert torch.equal(parent.new_codes.detach(),new_code_start(1201,'mirror',0,1024))


def test_meta_checkpoint_corruption_blocks_audit_before_scoring(tmp_path):
    from run import run_experiment
    cfg=dict(experiment='A12_SMOKE',families=['tent'],worlds=[1202],seeds=[0],methods=['shift'],cap=1024,
             meta_checkpoints=[2,4],support_counts=[32,256],adapt_checkpoints=[2,4],batch=16,workers=1)
    out=tmp_path/'tamper-meta'
    lock=run_experiment(out,cfg)
    p=out/lock['meta_models'][0]['path']
    raw=bytearray(p.read_bytes());raw[50]^=1;p.write_bytes(raw)
    with pytest.raises(RuntimeError,match='meta checkpoint hash'):
        audit_run(out)


def test_attribution_gate_counts_with_guardrails():
    from analyze import diagnostic_gates
    values=[]
    for world,improve,old_delta in [(1,.08,.08),(2,.07,.04),(3,.01,.20)]:
        for axis in ('meta','support','adapt'):
            values.append(dict(family='tent',method='mirror',axis=axis,world=world,
                delta_iid=improve,relative_iid_gain=0.10,delta_old=old_delta,delta_ood=.03))
    r=diagnostic_gates(values)
    assert next(z for z in r if z['axis']=='meta')['n_benefit']==1
    assert next(z for z in r if z['axis']=='support')['n_benefit']==2
    assert next(z for z in r if z['axis']=='adapt')['n_benefit']==2
