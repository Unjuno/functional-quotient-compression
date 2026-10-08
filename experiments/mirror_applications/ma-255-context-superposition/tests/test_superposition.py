import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import run_experiment as exp

def test_pair_rotation_inverse():
    x=torch.randn(4,exp.D);theta=torch.randn(4)
    assert float((exp.rotate(exp.rotate(x,theta[:,None]),-theta[:,None])-x).abs().max())<1e-6

def test_phase_binding_unbinding_is_identity_for_single_task():
    w=torch.randn(1,exp.D);theta=torch.tensor([1.2])
    got=exp.retrieve_phase(exp.build_phase(w,theta),theta)
    assert float((got-w).abs().max())<1e-6

def test_world_is_deterministic_and_task_labels_differ():
    t1,d1=exp.make_world(25501);t2,d2=exp.make_world(25501)
    assert torch.equal(t1,t2)
    assert torch.equal(d1['test'][0],d2['test'][0])
    assert d1['test'][0].shape==(exp.T,4096,exp.D)
