from pathlib import Path
import sys
import numpy as np
SRC=Path(__file__).resolve().parents[1]/'source';sys.path.insert(0,str(SRC))
from engine import D,K,T,FRACTIONS,make_world,make_state,decode,save_payload,load_payload

def test_private_fraction_and_basis_contract():
    _,basis,private,_,_,_,*_=make_world(31501,0.25)
    assert basis.shape==(D,K) and T==256 and private.sum()==64
    np.testing.assert_allclose(basis.T@basis,np.eye(K),atol=2e-6)

def test_common_orbit_is_recovered_without_private_coordinates():
    theta,basis,_,_,_,targets,xtr,xval,_,ytr,yval,_=make_world(31501,0.0)
    state,_=make_state('mirror_sparse',theta,basis,xtr,ytr,xval,yval)
    assert np.all(state['dims']==0)
    got=np.stack([decode('mirror_sparse',state,t) for t in range(T)])
    assert np.mean((got-targets)**2)<1e-5

def test_private_tasks_receive_residual_dimensions():
    theta,basis,private,_,_,targets,xtr,xval,_,ytr,yval,_=make_world(31502,0.25)
    state,_=make_state('mirror_sparse',theta,basis,xtr,ytr,xval,yval)
    assert np.all(state['dims'][~private]==0)
    assert np.mean(state['dims'][private])>0
    got=np.stack([decode('mirror_sparse',state,t) for t in range(T)])
    assert np.mean((got-targets)**2)<2e-3

def test_payload_roundtrip_and_bytes_stable(tmp_path):
    theta,basis,_,_,_,_,xtr,xval,_,ytr,yval,_=make_world(31501,0.0)
    state,_=make_state('mirror_sparse',theta,basis,xtr,ytr,xval,yval)
    a,b=tmp_path/'a.npz',tmp_path/'b.npz'
    assert save_payload(a,state)==save_payload(b,state)
    assert a.read_bytes()==b.read_bytes()
    loaded=load_payload(a)
    for k in state:np.testing.assert_array_equal(state[k],loaded[k])
