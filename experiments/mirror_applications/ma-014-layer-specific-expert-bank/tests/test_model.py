import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from engine import teacher,target
from model import LayerExpertBank,METHODS

def test_four_layer_mirror_exactly_represents_aligned_teacher():
 t=teacher('aligned',33);m=LayerExpertBank('mirror',3)
 with torch.no_grad():m.wi.copy_(t['wi']);m.wo.copy_(t['wo']);m.input_angle.copy_(t['ai']);m.output_angle.copy_(t['ao'])
 x=torch.randn(256,8);l=torch.arange(256)%4;e=torch.arange(256)%4
 assert torch.allclose(m(x,l,e),target(t,x,l,e),atol=1e-7)
def test_all_methods_have_gradients_and_single_expert_shapes():
 x=torch.randn(12,8);l=torch.arange(12)%4;e=torch.arange(12)%4
 for method in METHODS:
  m=LayerExpertBank(method,7);assert m(x,l,e).shape==(12,8);m(x,l,e).square().mean().backward();assert all(p.grad is not None for p in m.parameters())
def test_independent_bank_has_sixteen_full_experts():
 m=LayerExpertBank('independent',4);assert m.wi.shape==(4,4,16,8) and m.wo.shape==(4,4,8,16)
