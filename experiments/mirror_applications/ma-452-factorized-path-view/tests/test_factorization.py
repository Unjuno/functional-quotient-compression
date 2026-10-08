import sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_all_path_and_role_factors_are_seen_but_combinations_are_not():
    assert {p for p,_ in run.TRAIN_PAIRS}=={0,1,2,3}
    assert {r for _,r in run.TRAIN_PAIRS}=={0,1,2}
    assert set(run.TRAIN_PAIRS).isdisjoint(run.TEST_PAIRS)

def test_heldout_pairs_complete_cartesian_product():
    assert len(set(run.TRAIN_PAIRS+run.TEST_PAIRS))==12

def test_native_givens_and_mirror_function_are_same():
    a=torch.tensor([[.4,-.2,.1,.7],[.3,.2,-.5,.1]])
    b=torch.tensor([[.1,.5,-.2,.3],[-.1,.7,.4,-.3]])
    codebook=torch.tensor([[-.3,.6],[.2,-.5],[.7,.1]])
    mirror=run.model_weight('mirror',a,b,2,1,[codebook])
    native=run.model_weight('native_givens',a,b,2,1,[codebook])
    assert torch.equal(mirror,native)

def test_actual_payload_includes_index_arrays_and_schema():
    raw=run.payload_bytes({'route_ids':np.zeros((4,2),np.int8),'role_ids':np.zeros(4,np.int8)},{'v':1})
    d=np.load(__import__('io').BytesIO(raw));assert d['route_ids'].shape==(4,2);assert '__schema_json__' in d.files
