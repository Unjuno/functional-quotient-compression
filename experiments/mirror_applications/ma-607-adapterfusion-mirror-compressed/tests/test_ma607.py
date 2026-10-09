import sys,io
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_experiment import Teacher,Bank,pack,fit_fusion,D,O,R,K

def test_oracle_bank_matches_teacher():
    t=Teacher(2);b=Bank('oracle',3,t);x=torch.randn(9,D)
    assert torch.allclose(t.outputs(x),b.outputs(x),atol=1e-6,rtol=1e-6)

def test_mirror_payload_roundtrip():
    b=Bank('mirror',4);g=torch.nn.Linear(D,K);x=torch.randn(5,D);blob=pack(b,g,'mirror',4);state=torch.load(io.BytesIO(blob),weights_only=False)['state']
    c=Bank('mirror',4);c.load_state_dict(state['bank']);h=torch.nn.Linear(D,K);h.load_state_dict(state['fusion_gate'])
    assert torch.equal(b.outputs(x),c.outputs(x));assert torch.equal(g(x),h(x))

def test_fusion_gate_repeated_updates():
    t=Teacher(6);b=Bank('mirror',7)
    gate,_=fit_fusion(b,8,t,3)
    assert gate(torch.randn(2,D)).shape==(2,K)
