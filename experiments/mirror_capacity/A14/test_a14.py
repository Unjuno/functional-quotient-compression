import sys
from pathlib import Path
import hashlib
import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import a14
sys.path.insert(0,str(HERE.parent/'mirror_es_a13_20261115'))
from a13 import make_world,Student,serialize,deserialize


def test_oracle_inputs_have_both_fixed_splits_and_no_support():
    w=make_world(1200,'tent')
    x,y=a14.oracle_examples(w,2)
    assert x.shape==(512,1) and y.shape==(512,)
    assert torch.equal(x[:256],w['new_iid_x'])
    assert torch.equal(y[256:],w['new_ood_y'][:,2])
    assert not torch.equal(x[:32],w['support_x'][2])


def test_locked_model_validation_rejects_corruption(tmp_path):
    parent=Student('shift',20)
    child=deserialize(serialize(parent))
    digest=hashlib.sha256(serialize(parent)).hexdigest()
    a14.check_frozen_parent(parent,child,digest)
    with torch.no_grad():child.up.weight[0,0]+=0.01
    with pytest.raises(RuntimeError,match='shared'):
        a14.check_frozen_parent(parent,child,digest)


def test_oracle_reproducibility_and_bounded_outputs():
    m=Student('mirror',20).eval()
    w=make_world(1201,'tent')
    x,y=a14.oracle_examples(w,0)
    c=m.new_codes[0].detach().numpy()
    r1=a14.oracle_code(m,x,y,c,seed=11,pop=8,generations=3)
    r2=a14.oracle_code(m,x,y,c,seed=11,pop=8,generations=3)
    assert np.array_equal(r1['code'],r2['code'])
    assert r1['evaluations']==24
    assert np.all(np.abs(r1['code'])<=4.0)
    assert r1['oracle_mse']<=r1['incumbent_mse']+1e-12


def test_oracle_scorer_is_equal_to_pytorch_model():
    m=Student('shift',20).eval()
    w=make_world(1200,'tent')
    x,y=a14.oracle_examples(w,0)
    c=np.array([0.3,-0.2],dtype=np.float64)
    a14.verify_kernel(m,x,c)


def test_oracle_challenge_generator_matches_pinned_teacher():
    from posthoc import independent_challenge
    world=make_world(1301,'tent')
    _,_,_,_,_,_,teacher=independent_challenge(1301,n=32)
    synthesized=teacher(world['new_iid_x'].numpy())[:,16:]
    assert np.max(np.abs(synthesized-world['new_iid_y'].numpy()))<2e-6


def test_audit_rejects_wrong_config_hash_before_scoring(tmp_path):
    import json
    (tmp_path/'config.json').write_text(json.dumps({'source_sha256':{}}))
    (tmp_path/'train.lock.json').write_text(json.dumps({'config_sha256':'0'*64,'models':[]}))
    with pytest.raises(RuntimeError,match='frozen config'):
        a14.audit(tmp_path)


def test_repository_layout_uses_pinned_sibling_a13(tmp_path):
    mirror=(tmp_path/'mirror_capacity')
    a14dir=mirror/'A14';a13dir=mirror/'A13'
    a14dir.mkdir(parents=True);a13dir.mkdir()
    (a13dir/'a13.py').write_text('# pinned A13')
    assert a14.select_a13_root(a14dir)==a13dir
