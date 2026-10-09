import sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_path_router_encodes_two_layer_choice():
    assert len(run.ROUTES)==4 and run.ROUTES[3]==(1,1)

def test_givens_view_changes_selected_path_function():
    w=torch.tensor([1.,.4,.7,-.2]);z=torch.tensor([.3,-.5])
    assert not torch.equal(run.rotate(w,torch.zeros(2)),run.rotate(w,z))

def test_mirror_native_map_is_exact():
    w=torch.tensor([.2,.4,-.3,.8]);z=torch.tensor([.5,-.4])
    assert torch.equal(run.weight('mirror',w[None,:].repeat(2,1),w[None,:].repeat(2,1),(0,0),z,[]),run.weight('native_givens',w[None,:].repeat(2,1),w[None,:].repeat(2,1),(0,0),z,[]))

def test_payload_roundtrip_counts_metadata():
    raw=run.serialize({'route_ids':np.zeros((8,2),dtype=np.int8),'codes':np.ones((8,2),dtype=np.float32)},{'v':1})
    d=np.load(__import__('io').BytesIO(raw));assert d['route_ids'].shape==(8,2);assert '__schema_json__' in d.files
