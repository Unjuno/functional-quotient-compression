import sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parents[1]/'source';sys.path.insert(0,str(P))
from model import METHODS,ExpandedAttention,givens
from engine import make_world,target

def test_givens_preserves_norm():
 x=torch.randn(8,32);assert torch.allclose(x.norm(dim=-1),givens(x,torch.randn(16)).norm(dim=-1),atol=3e-6)

def test_all_variants_forward_backward_and_payload():
 x=torch.randn(2,6,32)
 for method in METHODS:
  m=ExpandedAttention(method,5);y=m(x);assert y.shape==(2,6,32) and torch.isfinite(y).all() and m.serialized_payload_bytes()>0;y.square().mean().backward()

def test_mirror_expansion_exactly_matches_aligned_teacher():
 w=make_world(72,'shared4_view16');x=torch.randn(2,6,32);m=ExpandedAttention('mirror_views')
 with torch.no_grad():m.w.copy_(w['w']);m.angle.copy_(w['angle']);m.out.copy_(w['out'])
 assert torch.allclose(m(x),target(x,w,'shared4_view16'),atol=4e-6)
