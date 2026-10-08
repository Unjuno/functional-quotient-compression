import importlib.util
from pathlib import Path
import numpy as np

P=Path(__file__).parents[1]/"source"/"run.py"
spec=importlib.util.spec_from_file_location("ma335",P); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_c4_orbit_and_stabilizer():
    W,R,targets,x=m.make_world(33511)
    assert np.allclose(targets[0],targets[2])
    assert np.allclose(targets[1],targets[3])
    assert not np.allclose(targets[0],targets[1])
    assert np.max(np.abs(np.mean([r@W@r.T for r in R],axis=0)-R[1]@np.mean([r@W@r.T for r in R],axis=0)@R[1].T))<1e-6

def test_serialized_roundtrip_and_paid_address():
    p,a,md=m.pack_payload({"base":np.eye(2,dtype=np.float32),"codes":np.arange(4,dtype=np.float32)}, {"kind":"group_action"})
    assert len(p)>0 and a["base"].shape==(2,2) and a["codes"].tolist()==[0,1,2,3]
    assert md["kind"]=="group_action"

def test_group_conjugation_reconstructs_and_needs_private_task():
    W,R,targets,x=m.make_world(33512)
    preds=[x@(r@W@r.T).T for r in R]
    for p,t in zip(preds,targets[:4]):
        assert np.max(np.abs(p-x@t.T))<1e-6
    shared=x@(W.T)
    assert np.mean((shared-x@targets[4].T)**2)>1e-4

def test_direct_irrep_control_matches_group_views():
    W,R,targets,x=m.make_world(33501)
    a=(W[0,0]+W[1,1])/2; b=(W[1,0]-W[0,1])/2
    c=(W[0,0]-W[1,1])/2; d=(W[0,1]+W[1,0])/2
    for k in range(4):
        parity=k%2; s=1 if parity==0 else -1
        decoded=np.array([[a+s*c,-b+s*d],[b+s*d,a-s*c]],dtype=np.float32)
        assert np.max(np.abs(decoded-R[k]@W@R[k].T))<1e-6
