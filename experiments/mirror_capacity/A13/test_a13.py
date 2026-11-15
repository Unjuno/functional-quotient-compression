from pathlib import Path
import sys
import hashlib
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str((ROOT.parent/'A11') if (ROOT.parent/'A11'/'a11.py').is_file() else (ROOT/'source/repo/experiments/mirror_capacity/A11')))
from a11 import make_model,make_world,deserialize,serialize,count_bytes,Student
from a13 import candidate_predict,es_optimize,random_optimize,make_combined_model,score_support,load_archived_parent

@pytest.mark.parametrize('method', ['mirror','shift','film'])
def test_candidate_forward_matches_torch(method):
    model=make_model(method,1024,seed=73)
    world=make_world(1200,'tent')
    sx=world['support_x'][0,:7]
    codes=np.array([[0.17,-0.33],[1.35,0.44]],dtype=np.float64)
    np_pred=candidate_predict(model,sx,codes)
    assert np_pred.shape==(2,7)
    with torch.no_grad():
        for i,c in enumerate(codes):
            model.new_codes[0].copy_(torch.tensor(c,dtype=torch.float32))
            t_pred=model(sx,torch.full((7,),16,dtype=torch.long)).numpy()
            np.testing.assert_allclose(np_pred[i],t_pred,atol=3e-5,rtol=2e-5)


def test_es_deterministic_and_population_budget():
    model=make_model('shift',1024,seed=31)
    world=make_world(1200,'tent')
    kwargs=dict(model=model,x=world['support_x'][0],y=world['support_y'][0],start=model.new_codes[0].detach().numpy(),seed=19,pop=8,generations=3)
    a=es_optimize(**kwargs)
    b=es_optimize(**kwargs)
    assert a['evaluations']==24
    np.testing.assert_array_equal(a['code'],b['code'])
    assert a['best_support_mse']==b['best_support_mse']
    assert len(a['trace'])==3


def test_random_search_uses_same_budget():
    model=make_model('mirror',1024,seed=8)
    world=make_world(1201,'sine')
    out=random_optimize(model,world['support_x'][0],world['support_y'][0],model.new_codes[0].detach().numpy(),seed=123,evaluations=24)
    assert out['evaluations']==24
    assert out['best_support_mse']<=score_support(model,world['support_x'][0],world['support_y'][0],model.new_codes[0].detach().numpy())+1e-12


def test_new_code_is_only_changed_tensor():
    m=make_model('mirror',1024,seed=7)
    original={k:v.clone().detach() for k,v in m.named_parameters()}
    codes=np.arange(16,dtype=np.float32).reshape(8,2)*.01
    changed=make_combined_model(m,codes)
    for key,v in changed.named_parameters():
        if key!='new_codes':assert torch.equal(v,original[key])
    assert count_bytes(changed)==count_bytes(m)
    assert serialize(deserialize(serialize(changed)))==serialize(changed)


def test_a12_archived_parent_hash_is_verified():
    with pytest.raises((ValueError, RuntimeError)):
        load_archived_parent(zip_file=ROOT/'missing.zip',family='tent',world_id=1200,seed=0,method='mirror')
