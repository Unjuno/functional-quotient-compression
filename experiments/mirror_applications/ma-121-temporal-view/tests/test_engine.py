import io,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from engine import PacketModel,SLOTS,D,VOCAB,make_world,payload

def test_packet_shapes_and_mirror_gradient():
 x,y,_,_=make_world(121,True);m=PacketModel('mirror_phase');z=m(x);loss=torch.nn.functional.cross_entropy(z.reshape(-1,VOCAB),y.reshape(-1));loss.backward()
 assert z.shape==(256,SLOTS,VOCAB) and torch.isfinite(m.angle.grad).all()

def test_packet_model_payload_roundtrip():
 m=PacketModel('ptp_rank2');state=torch.load(io.BytesIO(payload(m)),weights_only=False);copy=PacketModel(state['method']);copy.load_state_dict(state['state_dict'])
 x=torch.randn(4,D);assert torch.equal(m(x),copy(x))
