import sys,io
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_experiment import Teacher,Student,data,pack

def test_all_models_and_rotation():
    t=Teacher(1);x,y=data(2,t,7);assert y.shape==(7,10)
    for m in ('static','mirror','film','soft_moe'):assert Student(m,3)(x).shape==y.shape

def test_payload_roundtrip():
    m=Student('mirror',3);z=torch.randn(4,16);b=pack(m,'mirror',3);state=torch.load(io.BytesIO(b),weights_only=False)['state'];n=Student('mirror',3);n.load_state_dict(state);assert torch.equal(m(z),n(z))
