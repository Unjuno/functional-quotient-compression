import sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parents[1]/'source';sys.path.insert(0,str(P))
from model import METHODS,VirtualLoRA,task_id
from engine import make_world,target

def test_task_addresses_cover_all_eight():assert torch.unique(task_id(torch.randn(4096,16))).numel()==8

def test_all_adapter_methods_gradient_and_serialization():
 x=torch.randn(24,16)
 for method in METHODS:
  m=VirtualLoRA(method,11);y=m(x);assert y.shape==(24,12) and torch.isfinite(y).all() and m.serialized_payload_bytes()>0;y.square().mean().backward()

def test_mirror_rank_rotation_exactly_represents_teacher():
 w=make_world(4,'shared_rotated_lora');x=torch.randn(50,16);m=VirtualLoRA('mirror_lora')
 with torch.no_grad():m.a.copy_(w['a']);m.b.copy_(w['b']);m.angle.copy_(w['angle'])
 assert torch.allclose(m(x),target(x,w,'shared_rotated_lora'),atol=1e-6)
