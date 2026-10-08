from pathlib import Path
import sys

import numpy as np

SRC=Path(__file__).resolve().parents[1]/"source"
sys.path.insert(0,str(SRC))
from engine import D,K,T,LEVELS,make_world,make_state,decode_task,save_payload,load_payload

def test_world_has_balanced_varying_intrinsic_dimensions():
    _,basis,_,dims,_,*_=make_world(31401)
    assert basis.shape==(D,K)
    np.testing.assert_allclose(basis.T@basis,np.eye(K),atol=2e-6)
    assert T==48
    assert {d:int(np.sum(dims==d)) for d in LEVELS}=={2:16,4:16,8:16}

def test_adaptive_mirror_recovers_minimum_dimensions():
    theta,basis,radii,dims,targets,xtr,xval,xte,ytr,yval,yte=make_world(31401)
    state,_,_=make_state("adaptive_mirror",theta,basis,radii,xtr,ytr,xval,yval)
    assert np.mean(state["dims"]==dims)>=.95
    weights=np.stack([decode_task("adaptive_mirror",state,t) for t in range(T)])
    assert np.mean((weights-targets)**2)<2e-5

def test_private_direction_is_outside_shared_basis():
    _,basis,_,_,_,*_=make_world(31402)
    direction=np.random.default_rng(14).normal(size=D)
    direction-=basis@(basis.T@direction)
    # Any coordinate code on this basis has an irreducible off-basis residual.
    assert np.linalg.norm(direction)>0.1

def test_payload_roundtrip_is_exact_and_bytes_stable(tmp_path):
    theta,basis,radii,_,_,xtr,xval,_,ytr,yval,_=make_world(31403)
    state,_,_=make_state("adaptive_mirror",theta,basis,radii,xtr,ytr,xval,yval)
    a,b=tmp_path/"a.npz",tmp_path/"b.npz"
    assert save_payload(a,state)==save_payload(b,state)
    assert a.read_bytes()==b.read_bytes()
    loaded=load_payload(a)
    for key in state: np.testing.assert_array_equal(loaded[key],state[key])
