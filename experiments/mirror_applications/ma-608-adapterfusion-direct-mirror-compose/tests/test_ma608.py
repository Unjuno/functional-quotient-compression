import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_experiment import make,fused_reference,fused_direct,fused_full,D,K

def test_linear_code_composition_exact():
    a,b,codes,mats,gate=make(8);x=torch.randn(12,D)
    assert torch.allclose(fused_reference(x,a,b,mats,gate),fused_direct(x,a,b,mats,gate),atol=1e-6,rtol=1e-6)
    afull=torch.stack([mats[t]@a for t in range(K)])
    assert torch.allclose(fused_reference(x,a,b,mats,gate),fused_full(x,afull,b,gate),atol=1e-6,rtol=1e-6)

def test_nonlinear_case_is_not_assumed_exact():
    a,b,codes,mats,gate=make(9);x=torch.randn(128,D)
    r=fused_reference(x,a,b,mats,gate,True);d=fused_direct(x,a,b,mats,gate,True)
    assert ((d-r)**2).mean()>1e-7
