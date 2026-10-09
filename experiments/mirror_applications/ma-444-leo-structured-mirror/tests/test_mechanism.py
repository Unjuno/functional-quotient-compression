import sys
from pathlib import Path
import numpy as np
import torch
SOURCE=Path(__file__).resolve().parents[1]/'source';sys.path.insert(0,str(SOURCE))
import run

def test_rotation_zero_is_identity_and_changes_function():
    z=torch.zeros(3)
    assert torch.equal(run.rotate(run.TEMPLATE,z),run.TEMPLATE)
    assert torch.linalg.vector_norm(run.rotate(run.TEMPLATE,torch.tensor([.2,-.3,.4]))-run.TEMPLATE)>0.1

def test_native_givens_is_exact_mirror_alias():
    base=run.TEMPLATE+torch.arange(9,dtype=torch.float32)/100
    z=torch.tensor([.3,-.5,.2])
    assert torch.equal(run.decode('mirror',base,z,[]),run.decode('native_givens',base,z,[]))

def test_npz_payload_counts_arrays_and_metadata():
    payload=run.pack({'base':np.ones(9,dtype=np.float32),'codes':np.zeros((16,3),dtype=np.float32)},{'version':1})
    loaded=np.load(__import__('io').BytesIO(payload))
    assert np.array_equal(loaded['base'],np.ones(9,dtype=np.float32))
    assert loaded['__schema_json__'].dtype==np.uint8
    assert len(payload)>9*4+16*3*4
