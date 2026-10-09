import io,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from engine import RopeModel,TEST_POS,world,payload

def test_rope_shapes_and_finite_gradient():
 m=RopeModel('mirror_rope');out=m(torch.arange(16,dtype=torch.float32));out.square().mean().backward()
 assert out.shape==(4,4,16,2) and torch.isfinite(m.address.grad).all()

def test_actual_rope_payload_roundtrip():
 m=RopeModel('mirror_rope');artifact=torch.load(io.BytesIO(payload(m)),weights_only=False)
 copy=RopeModel(artifact['method']);copy.load_state_dict(artifact['state_dict'])
 assert torch.equal(m(TEST_POS),copy(TEST_POS))
 assert world(116)[1].shape==(4,4,112,2)
