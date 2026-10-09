import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from engine import teacher,target
from model import DomainExperts,METHODS
def test_domain_view_matches_aligned_teacher():
 t=teacher('aligned',4);m=DomainExperts('mirror',8)
 with torch.no_grad():m.wi.copy_(t['wi']);m.wo.copy_(t['wo']);m.input_angle.copy_(t['ai']);m.output_angle.copy_(t['ao'])
 x=torch.randn(120,16);d=torch.arange(120)%3;e=torch.arange(120)%4;assert torch.allclose(m(x,d,e),target(t,x,d,e),atol=1e-7)
def test_all_controls_are_differentiable():
 x=torch.randn(12,16);d=torch.arange(12)%3;e=torch.arange(12)%4
 for method in METHODS:
  m=DomainExperts(method,9);m(x,d,e).square().mean().backward();assert all(p.grad is not None for p in m.parameters())
def test_independent_pool_has_twelve_full_functions():
 m=DomainExperts('independent',7);assert m.wi.shape==(3,4,32,16) and m.wo.shape==(3,4,16,32)
