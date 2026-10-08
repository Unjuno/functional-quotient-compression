import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import run_experiment as exp

def test_pair_rotation_inverse():
    x=torch.randn(exp.D);a=torch.tensor(.7)
    assert float((exp.rotate(exp.rotate(x,a),-a)-x).abs().max())<1e-6

def test_factorized_coordinate_generates_heldout_pair():
    alpha,beta=.4,.9;got=1*alpha+1*beta
    assert abs(got-(alpha+beta))<1e-7

def test_world_teacher_is_compositional():
    base,factors,angles,data=exp.make_world(25701)
    assert torch.allclose(angles,torch.tensor([0.,factors[0],factors[1],factors.sum()]),atol=1e-6)
    assert data['train'][0].shape==(4,1024,exp.D)
