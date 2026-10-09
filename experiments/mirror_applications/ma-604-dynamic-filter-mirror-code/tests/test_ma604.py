import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_experiment import Teacher,Student,data,pack

def test_method_shapes_and_givens_norm():
    t=Teacher(3); x,y=data(4,t,9)
    assert y.shape==(9,8)
    for method in ('static','mirror','basis','full_filter'):
        m=Student(method,5); assert m(x).shape==y.shape
    q=Student('mirror',5); assert torch.isfinite(q(x)).all()

def test_payload_replay():
    t=Teacher(3); a=Student('mirror',5); blob=pack(a,'mirror',5); state=torch.load(__import__('io').BytesIO(blob),weights_only=False)['state']; b=Student('mirror',5); b.load_state_dict(state)
    x=torch.randn(6,12); assert torch.equal(a(x),b(x))
