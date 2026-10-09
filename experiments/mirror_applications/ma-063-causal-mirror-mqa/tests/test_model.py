import sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parents[1]/'source';sys.path.insert(0,str(P))
from model import METHODS,CausalKV,rotate_heads,cache_bytes
from engine import make_world,target

def test_vectorized_rotation_preserves_norm_and_cache_cost():
 z=torch.randn(2,4,16,4);a=torch.randn(4,2);r=rotate_heads(z,a);assert torch.allclose(z.norm(dim=-1),r.norm(dim=-1),atol=2e-6)
 assert cache_bytes('mirror_kv')==cache_bytes('mqa')<cache_bytes('gqa2')<cache_bytes('mha')

def test_all_variants_backward_and_serialize():
 x=torch.randn(2,16,16)
 for method in METHODS:
  m=CausalKV(method,4);y=m(x);assert y.shape==(2,16,16) and torch.isfinite(y).all() and m.serialized_payload_bytes()>0;y.square().mean().backward()

def test_future_tokens_do_not_change_past_output():
 torch.manual_seed(2);m=CausalKV('mirror_kv');x=torch.randn(1,16,16);z=x.clone();z[:,10:]+=10
 assert torch.allclose(m(x)[:,:10],m(z)[:,:10],atol=1e-6)

def test_mirror_matches_shared_causal_teacher():
 w=make_world(31,'shared_causal_kv');x=torch.randn(1,16,16);m=CausalKV('mirror_kv')
 with torch.no_grad():m.q.copy_(w['q']);m.k.copy_(w['k']);m.v.copy_(w['v']);m.ka.copy_(w['ka']);m.va.copy_(w['va']);m.out.copy_(w['out'])
 assert torch.allclose(m(x),target(x,w,'shared_causal_kv'),atol=2e-6)
