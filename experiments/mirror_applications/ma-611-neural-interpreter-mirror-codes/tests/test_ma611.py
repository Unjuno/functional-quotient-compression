import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_experiment import basis,mirror_predict,coeff_predict,target

def test_phase_view_matches_coefficients():
    x=torch.linspace(-3,3,33);theta=torch.tensor(.73);coeff=torch.stack((theta.cos(),theta.sin()))
    assert torch.allclose(mirror_predict(x,2,theta),coeff_predict(x,2,coeff),atol=1e-6,rtol=1e-6)
    assert torch.allclose(mirror_predict(x,2,theta),target(x,2,theta),atol=1e-6,rtol=1e-6)

def test_signature_selects_frequency_basis():
    x=torch.linspace(-2,2,11)
    for s,w in enumerate((1.,2.,3.,4.)):
        z=basis(x,s);assert torch.allclose(z[:,0],torch.sin(w*x));assert torch.allclose(z[:,1],torch.cos(w*x))
