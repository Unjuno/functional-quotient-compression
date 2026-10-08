import io,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from engine import DEPTH,WIDTH,GroupModel,make_world,payload

def test_group_models_expand_to_logical_depth():
 for method in ['tie2','mirror2','tie4','mirror4','untied']:
  m=GroupModel(method);assert m.matrices().shape==(DEPTH,WIDTH,WIDTH)

def test_aligned_teacher_deterministic_and_payload_serializes():
 a=make_world(86,True);b=make_world(86,True)
 assert all(torch.equal(x,y) for x,y in zip(a,b))
 model=GroupModel('mirror2');artifact=torch.load(io.BytesIO(payload(model)),weights_only=False)
 restored=GroupModel(artifact['method']);restored.load_state_dict(artifact['state_dict'])
 x=torch.randn(2,WIDTH);assert torch.equal(model(x),restored(x))
