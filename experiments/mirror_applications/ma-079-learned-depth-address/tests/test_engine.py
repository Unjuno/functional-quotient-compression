import io,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from engine import AddressModel,DEPTH,WIDTH,make_world,payload_bytes

def test_depth_address_builds_all_logical_maps_and_receives_gradient():
 m=AddressModel('mirror_address');m(torch.randn(4,WIDTH)).square().mean().backward()
 assert m.matrices().shape==(DEPTH,WIDTH,WIDTH)
 assert m.angle_coeff.grad is not None and torch.isfinite(m.angle_coeff.grad).all()

def test_unseen_layer_address_is_a_function_of_depth_and_state_serializes():
 a=AddressModel('mirror_address'); payload=payload_bytes(a)
 assert len(payload)>0
 with torch.no_grad(): a.angle_coeff.copy_(torch.tensor([0.7,0.25]))
 mats=a.matrices().detach()
 assert not torch.equal(mats[1],mats[6])
 artifact=torch.load(io.BytesIO(payload_bytes(a)),weights_only=False)
 restored=AddressModel(artifact['method']);restored.load_state_dict(artifact['state_dict'])
 x=torch.randn(2,WIDTH)
 assert torch.equal(a(x),restored(x))
 x,y,xt,yt=make_world(79,True)
 assert y.shape==(DEPTH,256,WIDTH) and yt.shape==(DEPTH,128,WIDTH)
