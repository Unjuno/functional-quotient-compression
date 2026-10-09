import sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parents[1]/'source';sys.path.insert(0,str(P))
from model import METHODS,RoutedExperts,role_of,givens
from engine import make_world,target,draw_x

def test_shift_makes_role_zero_rare():
 x=draw_x(50000,2,True);freq=float((role_of(x)==0).float().mean());assert .008<freq<.02

def test_givens_norm_preserving():
 x=torch.randn(16,16);a=torch.randn(8);assert torch.allclose(x.norm(dim=-1),givens(x,a).norm(dim=-1),atol=2e-6)

def test_all_models_shape_finite_and_serializable():
 x=torch.randn(12,16)
 for method in METHODS:
  m=RoutedExperts(method,4);y=m(x);assert y.shape==(12,12) and torch.isfinite(y).all() and m.serialized_payload_bytes()>0
  y.square().mean().backward()

def test_private_mirror_can_represent_aligned_teacher():
 w=make_world(19,'shared_common_private_rare');x=draw_x(128,77,True);m=RoutedExperts('mirror_private_rare')
 with torch.no_grad():m.shared.copy_(w['shared']);m.rare.copy_(w['rare']);m.angles.copy_(w['angles'][1:])
 assert torch.allclose(m(x),target(x,w,'shared_common_private_rare'),atol=2e-6)
