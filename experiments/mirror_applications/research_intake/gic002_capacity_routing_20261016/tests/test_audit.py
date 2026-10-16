import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import pytest

def test_independent_numpy_inference(tmp_path):
    import engine, audit
    torch.set_num_threads(1)
    rng=np.random.default_rng(0);x=rng.normal(size=(31,5)).astype('f4');c=rng.normal(size=(31,6)).astype('f4')
    for kind in ['mirror','gain','independent']:
        m=engine.Bank(kind,8,[3]);p=tmp_path/(kind+'.bin');engine.save(m,0,p)
        expected=m(torch.from_numpy(x),torch.from_numpy(c))[0][0].detach().numpy()
        got=audit.numpy_predict(p,x,c)
        assert np.max(np.abs(got-expected))<2e-6

def test_missing_payload_fails(tmp_path):
    import audit
    with pytest.raises(FileNotFoundError):audit.numpy_predict(tmp_path/'absent',np.zeros((1,5)),np.zeros((1,6)))

def test_summary_world_unit():
    import analyze
    rows=[]
    for w in [1,2]:
        for i,t in enumerate([1.,3.]):
            rows.append(dict(world=w,family='aligned',ntrain=64,method='mirror1_free',init=i,test_mse=t,train_mse=0.,worst_mse=t,usage=.5,expert_difference=1.,route_match=.5,bytes=1256,best_test_mse=t))
    ans=analyze.world_summary(rows)
    assert len(ans)==2
    assert ans[0]['test_mean']==2. and ans[0]['seed_variance']==2.


def test_widen_preserves_parent_function():
    import engine,widen
    parent=engine.Bank('mirror',1,[0,1]);expanded=widen.branch(parent,'expand8zero')
    x=torch.randn(17,5);c=torch.randn(17,6)
    assert torch.allclose(parent(x,c)[0],expanded(x,c)[0],atol=1e-6)
    assert torch.equal(expanded.codes[:,:,1:],torch.zeros_like(expanded.codes[:,:,1:]))


def test_zero_added_angles_allow_nonzero_gradients():
    import engine,widen
    torch.manual_seed(23)
    parent=engine.Bank('mirror',1,[0]);m=widen.branch(parent,'expand8zero')
    x=torch.randn(31,5);c=torch.randn(31,6);target=torch.randn(31)
    loss=(m(x,c)[0]-target).square().mean();loss.backward()
    assert m.codes.grad[:,:,1:].abs().sum().item()>1e-6
