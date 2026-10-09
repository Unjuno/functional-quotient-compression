import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_experiment import make,target,full_codes,decode_full,decode_mirror,N

def test_full_mirror_residual_exact():
    sig,phase,_,psi,amp=make(4);private=torch.arange(N)<16;x=torch.linspace(-2,2,23);y=target(x,sig,phase,private,psi,amp);full=decode_full(x,full_codes(sig,phase,private,psi,amp));idx=torch.where(private)[0];c=torch.stack((amp*psi[idx].cos(),amp*psi[idx].sin()),1);mir=decode_mirror(x,sig,phase,idx,c)
    assert torch.allclose(full,y,atol=1e-6,rtol=1e-6);assert torch.allclose(mir,y,atol=1e-6,rtol=1e-6)

def test_zero_residual_is_shared_phase_family():
    sig,phase,_,psi,amp=make(5);x=torch.linspace(-2,2,19);y=target(x,sig,phase,torch.zeros(N,dtype=torch.bool),psi,amp);p=decode_mirror(x,sig,phase,torch.empty(0,dtype=torch.long),torch.empty(0,2))
    assert torch.allclose(p,y,atol=1e-6,rtol=1e-6)
