import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run import basis,rotate,encode,decode,world

def test_random_projection_and_rotation_geometry():
    b=basis(31101);z=np.arange(32,dtype=np.float64)/32
    assert b.shape==(128,32) and np.max(np.abs(b.T@b-np.eye(32)))<1e-12
    assert abs(np.linalg.norm(rotate(z,.13))-np.linalg.norm(z))<1e-10

def test_all_payload_state_kinds_decode():
    seed=31102;b,base,_=world(seed);u=np.linspace(-.2,.2,32);priv=np.linspace(.1,.3,32);full=np.linspace(-.3,.4,128)
    records=[{'kind':'base'},{'kind':'anchor'},{'kind':'angle','code':12},{'kind':'intrinsic'},{'kind':'full'}]
    blob=encode('mirror',seed,base,u,records,{3:priv},{4:full});v,_,_=decode(blob)
    assert v.shape==(5,128) and np.isfinite(v).all() and len(blob)>0
    assert np.allclose(v[0],base,atol=1e-7,rtol=1e-6)
    assert np.allclose(v[1],base+b@u.astype(np.float16).astype(np.float32),atol=1e-6)
    assert np.allclose(v[3],base+b@priv.astype(np.float16).astype(np.float32),atol=1e-6)
    assert np.allclose(v[4],full.astype(np.float32),atol=1e-7)

def test_world_task_splits():
    _,_,tasks=world(31103)
    assert len(tasks)==8
    assert tasks[0][0][0].shape==(512,128) and tasks[0][1][0].shape==(256,128) and tasks[0][2][0].shape==(512,128)
    assert not np.array_equal(tasks[0][0][0][:128],tasks[0][1][0][:128])
