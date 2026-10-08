import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import run_experiment as exp

def test_rotation_preserves_parameter_norm_and_is_invertible():
    x=torch.randn(exp.P);a=torch.tensor(.53);y=exp.rotate(x,a)
    assert abs(float(x.norm()-y.norm()))<1e-5
    assert float((exp.rotate(y,-a)-x).abs().max())<1e-6

def test_aligned_bank_is_rank_two():
    weights,_=exp.make_world(25821,'aligned');mean,basis,coeff=exp.fit_svd(weights);recon=mean+coeff@basis
    assert float((recon-weights.reshape(exp.E,exp.P)).square().mean())<1e-10

def test_unrelated_world_is_deterministic():
    w1,_=exp.make_world(25821,'unrelated');w2,_=exp.make_world(25821,'unrelated')
    assert torch.equal(w1,w2)
