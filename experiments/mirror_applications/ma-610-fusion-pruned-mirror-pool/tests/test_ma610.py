import sys,io
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_experiment import Teacher,Student,D,K,get_state,pack,predict_pruned

def test_teacher_and_pool_shapes():
    t=Teacher(3);g=torch.Generator().manual_seed(4);x=torch.randn(10,D,generator=g)
    assert t(x).shape==(10,10)
    for mode in ('mirror','full'):
        m=Student(mode,K,5);assert m(x).shape==(10,10)

def test_pruned_payload_replay():
    m=Student('mirror',K,6);ids=[0,2,4,6];s=get_state(m,ids);blob=pack('mirror',s,6,'test');load=torch.load(io.BytesIO(blob),weights_only=False)['state'];x=torch.randn(7,D)
    assert torch.equal(predict_pruned('mirror',s,x),predict_pruned('mirror',load,x))
