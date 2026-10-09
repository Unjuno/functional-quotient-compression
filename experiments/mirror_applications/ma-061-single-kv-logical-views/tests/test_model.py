import sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parents[1]/'source';sys.path.insert(0,str(P))
from model import METHODS,KVAttention,rot,cache_bytes
from engine import make_world,target

def test_rotation_norm_and_cache_scaling():
 x=torch.randn(7,4);assert torch.allclose(x.norm(dim=-1),rot(x,torch.randn(2)).norm(dim=-1),atol=2e-6)
 assert cache_bytes('mqa')==cache_bytes('mirror_kv')<cache_bytes('gqa2')<cache_bytes('mha')

def test_methods_forward_backward_and_serialize():
 x=torch.randn(3,8,16)
 for method in METHODS:
  m=KVAttention(method,9);y=m(x);assert y.shape==(3,8,16) and torch.isfinite(y).all() and m.serialized_payload_bytes()>0;y.square().mean().backward()

def test_mirror_kv_exactly_matches_shared_view_teacher():
 w=make_world(31,'shared_kv_views');x=torch.randn(2,8,16);m=KVAttention('mirror_kv')
 with torch.no_grad():m.q.copy_(w['q']);m.k.copy_(w['k']);m.v.copy_(w['v']);m.ka.copy_(w['ka']);m.va.copy_(w['va']);m.out.copy_(w['out'])
 assert torch.allclose(m(x),target(x,w,'shared_kv_views'),atol=2e-6)
