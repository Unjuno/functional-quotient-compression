"""Only mathematical/harness tests; no natural checkpoint or model evidence."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from orbit_audit import (canonical_delta_svd, projector, principal_cosines,
                         gauge_diagnostic, fit_shared_uv, projection_error,
                         screen_manifest)


def test_nonorthogonal_gauge_invariance():
    rng=np.random.default_rng(51)
    B=rng.normal(size=(11,3)); A=rng.normal(size=(3,9))
    G=np.array([[2.,.9,.4],[.1,1.4,-.6],[.3,.15,.8]])
    d=gauge_diagnostic(B,A,G)
    assert max(d.values())<1e-12
    U,s,V=canonical_delta_svd(B,A)
    assert np.allclose((U*s)@V.T,B@A,atol=1e-12)


def test_related_vs_unrelated_oracle_baseline():
    rng=np.random.default_rng(11)
    u=np.linalg.qr(rng.normal(size=(20,3)))[0]; v=np.linalg.qr(rng.normal(size=(15,3)))[0]
    train=[u@rng.normal(size=(3,3))@v.T for _ in range(6)]
    Ub,Vb=fit_shared_uv(train,3)
    related=u@rng.normal(size=(3,3))@v.T
    unrelated=rng.normal(size=(20,15))
    assert projection_error(related,Ub,Vb,'full')<1e-12
    assert projection_error(unrelated,Ub,Vb,'full')>.5
    assert projection_error(related,Ub,Vb,'diagonal')>=projection_error(related,Ub,Vb,'full')-1e-12
    assert projection_error(related,Ub,Vb,'sparse',sparse_q=3)>=projection_error(related,Ub,Vb,'full')-1e-12


def test_projector_and_angle_edge_cases():
    a=np.eye(5)[:,:2];b=np.eye(5)[:,1:3]
    assert np.allclose(principal_cosines(a,b),[1,0],atol=1e-12)
    assert np.allclose(projector(a)@a,a)
    with pytest.raises(ValueError): fit_shared_uv([np.eye(4)],5)
    with pytest.raises(ValueError): projection_error(np.eye(4),np.eye(4),np.eye(4),'sparse',-1)


def test_manifest_separation_and_base_revision(tmp_path):
    rng=np.random.default_rng(52)
    for name in ['train','audit']:
        np.savez(tmp_path/f'{name}.npz',B=rng.normal(size=(8,2)),A=rng.normal(size=(2,6)))
    import json
    data={'base_revision':'abc123','layer':'layers.0.q_proj','items':[
     {'task':'t1','split':'train','file':'train.npz','base_revision':'abc123'},
     {'task':'t2','split':'audit','file':'audit.npz','base_revision':'abc123'}]}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(data))
    report=screen_manifest(path,2,2)
    assert report['train_tasks']==1 and report['audit_tasks']==1
    assert report['results'][0]['full_core_error']>=0
    data['items'][1]['base_revision']='wrong';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='revision'):screen_manifest(path,2,2)
