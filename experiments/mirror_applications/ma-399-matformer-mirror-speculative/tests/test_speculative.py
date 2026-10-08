import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import run_experiment as exp

def test_nested_prefix_logits():
    model=exp.SuperNet();x=exp.onehot()[:3]
    assert model(x,16).shape==(3,exp.V)
    assert model(x,32).shape==(3,exp.V)
    assert model(x,64).shape==(3,exp.V)

def test_exact_rejection_correction_identity():
    pd=torch.softmax(torch.randn(64,exp.V),-1);pv=torch.softmax(torch.randn(64,exp.V),-1)
    corrected=exp.exact_corrected(pd,pv)
    assert float((corrected-pv).abs().max())<1e-6

def test_mirror_coordinate_and_basis_fit():
    model=exp.SuperNet();views=exp.fit_views(model)
    p=exp.distributions(model,torch.ones(exp.S,exp.V)/exp.V,views)
    assert p['mirror'].shape==(exp.S,exp.V)
    assert views[1].shape==(exp.S,)
    assert views[0].shape==(2,exp.V)

