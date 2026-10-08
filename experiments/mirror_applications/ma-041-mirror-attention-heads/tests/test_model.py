import sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parents[1]/'source';sys.path.insert(0,str(P))
from model import METHODS,AttentionHeads,givens
from engine import make_world,target

def test_givens_preserves_norm():
 x=torch.randn(8,16);assert torch.allclose(x.norm(dim=-1),givens(x,torch.randn(8)).norm(dim=-1),atol=2e-6)

def test_all_attention_variants_forward_backward_and_serialize():
 x=torch.randn(3,8,16)
 for method in METHODS:
  m=AttentionHeads(method,3);y=m(x);assert y.shape==(3,8,16) and torch.isfinite(y).all() and m.serialized_payload_bytes()>0;y.square().mean().backward()

def test_mirror_attention_exactly_matches_aligned_teacher():
 w=make_world(77,'shared_qkv_views');x=torch.randn(2,8,16);m=AttentionHeads('mirror_view')
 with torch.no_grad():m.w.copy_(w['w']);m.angle.copy_(w['angle']);m.out.copy_(w['out'])
 assert torch.allclose(m(x),target(x,w,'shared_qkv_views'),atol=2e-6)
