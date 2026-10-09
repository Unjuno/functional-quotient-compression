import sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parents[1]/'source';sys.path.insert(0,str(P))
from model import METHODS,ExpertBasis,role_of
from engine import make_world,target

def test_role_routing_covers_eight_balanced_cells():
 x=torch.randn(8192,16);assert torch.unique(role_of(x)).numel()==8

def test_all_models_shape_grad_and_serialization():
 x=torch.randn(16,16)
 for method in METHODS:
  m=ExpertBasis(method,4);y=m(x);assert y.shape==(16,12) and torch.isfinite(y).all() and m.serialized_payload_bytes()>0;y.square().mean().backward()

def test_mirror_angle_exactly_represents_unit_circle_teacher():
 w=make_world(12,'unit_circle_basis');x=torch.randn(64,16);m=ExpertBasis('mirror_angle')
 with torch.no_grad():m.basis.copy_(w['basis']);m.angle.copy_(w['coef'])
 assert torch.allclose(m(x),target(x,w,'unit_circle_basis'),atol=1e-6)
