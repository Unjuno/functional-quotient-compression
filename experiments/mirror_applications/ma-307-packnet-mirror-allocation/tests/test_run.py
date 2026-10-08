import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run import world,rotate,encode,decode

def test_payload_roundtrip_and_actual_bytes():
    _,base,_=world(30701)
    private=np.linspace(-.1,.1,32)
    records=[{'kind':'shared'},{'kind':'view','code':12},{'kind':'mask','mask':[1]*32},{'kind':'private'}]
    blob=encode('mirror',base,records,{3:private},30701);vectors,cfg=decode(blob)
    assert vectors.shape==(4,32) and len(blob)>0
    assert np.allclose(vectors[0],base,atol=1e-7,rtol=1e-6)
    assert np.allclose(vectors[1],rotate(np.asarray(base,dtype=np.float32),12*.2/127),atol=1e-7,rtol=1e-6)
    assert np.allclose(vectors[2],base,atol=1e-7,rtol=1e-6)
    assert np.allclose(vectors[3],private,atol=1e-7,rtol=1e-6)

def test_allocation_audit_retains_previous_functions():
    _,_,data=world(30704)
    from run import evaluate_stream
    _,events,_=evaluate_stream('mirror',30704,data,0.001)
    assert max(e['max_prior_task_n_mse_change'] for e in events)==0

def test_givens_preserves_norm():
    _,base,_=world(30702)
    assert abs(np.linalg.norm(rotate(base,.13))-np.linalg.norm(base))<1e-10

def test_world_split_and_sequence():
    _,_,data=world(30703)
    assert len(data)==8 and all(d[0][0].shape==(512,32) and d[1][0].shape==(256,32) and d[2][0].shape==(512,32) for d in data)
