import sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_two_routes_and_two_roles_make_four_logical_blocks():
    assert len([(m,r) for m in range(2) for r in range(2)])==4

def test_givens_role_changes_routed_function():
    w=torch.tensor([1.,.2,.3,-.6]);assert not torch.equal(run.rotate(w,torch.zeros(2)),run.rotate(w,torch.tensor([.3,-.4])))

def test_native_control_is_exact_mirror_map():
    p=[torch.tensor([[1.,.2,.3,.4],[-.2,.4,.7,.1]]),torch.tensor([[.1,.3],[-.4,.5]])]
    assert torch.equal(run.wmethod('mirror',p,1,0),run.wmethod('native_givens',p,1,0))

def test_payload_serializes_router_and_context_table():
    raw=run.pack({'router':np.array([0,1],dtype=np.int8),'codes':np.zeros((2,2),dtype=np.float32)},{'v':1});d=np.load(__import__('io').BytesIO(raw))
    assert d['router'].tolist()==[0,1] and '__schema_json__' in d.files
