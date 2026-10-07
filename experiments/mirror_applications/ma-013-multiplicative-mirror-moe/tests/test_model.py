import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from engine import teacher,target
from model import MultiplicativeExperts,METHODS

def test_product_scale_mirror_matches_aligned_teacher():
 t=teacher('aligned',4);m=MultiplicativeExperts('multiplicative',8)
 with torch.no_grad():m.wi.copy_(t['wi']);m.wo.copy_(t['wo']);m.u.copy_(t['u']);m.v.copy_(t['v']);m.alpha.copy_(t['alpha']);m.beta.copy_(t['beta'])
 x=torch.randn(20,16);r=torch.arange(20)%4;assert torch.allclose(m(x,r),target(t,x,r),atol=1e-7)
def test_methods_have_gradients():
 x=torch.randn(8,16);r=torch.arange(8)%4
 for method in METHODS:
  m=MultiplicativeExperts(method,9);m(x,r).square().mean().backward();assert all(p.grad is not None for p in m.parameters())
def test_factorized_scale_uses_compact_basis():
 m=MultiplicativeExperts('multiplicative',1);assert m.u.numel()+m.v.numel()+m.alpha.numel()+m.beta.numel()==68
