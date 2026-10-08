import io,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from engine import BRANCHES,D,Verifier,payload,world

def test_mirror_verifier_outputs_each_branch_and_has_gradient():
 x,y,_,_=world(129,True);m=Verifier('mirror_view');z=m(x);torch.nn.functional.binary_cross_entropy_with_logits(z,y).backward()
 assert z.shape==(512,BRANCHES) and torch.isfinite(m.angle.grad).all()

def test_serialized_verifier_roundtrip():
 m=Verifier('ptp_rank2');a=torch.load(io.BytesIO(payload(m)),weights_only=False);copy=Verifier(a['method']);copy.load_state_dict(a['state_dict']);x=torch.randn(3,D)
 assert torch.equal(m(x),copy(x))
