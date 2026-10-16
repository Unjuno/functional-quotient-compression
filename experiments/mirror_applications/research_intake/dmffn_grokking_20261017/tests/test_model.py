import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import Config, MirrorSwiGLU, mod_dataset, checkpoint_bytes

def test_four_functions_and_integer_labels():
  x,y=mod_dataset(17)
  assert x.shape==(289,34)
  assert y.shape==(289,4)
  a,b=5,7
  row=a*17+b
  assert y[row].tolist()==[(a+b)%17,(a+2*b)%17,(2*a+b)%17,(2*a+2*b)%17]

def test_methods_shape_finite_and_identity():
  cfg=Config(p=17,h=32,q=8)
  x,_=mod_dataset(17)
  for method in ('tied','gain','mirror','pair_mirror','independent'):
    torch.manual_seed(112)
    m=MirrorSwiGLU(cfg,method)
    out=m(x[:13]);assert out.shape==(13,4,17)
    assert torch.isfinite(out).all()
    assert checkpoint_bytes(m)>0
  torch.manual_seed(12);m=MirrorSwiGLU(cfg,'mirror');v=m(x[:13])
  assert torch.allclose(v[:,0],v[:,1],atol=0,rtol=0)

def test_mirror_rotation_preserves_squared_norm():
  cfg=Config(p=17,h=32,q=8)
  m=MirrorSwiGLU(cfg,'mirror')
  with torch.no_grad():m.ang_g.uniform_(-2,2)
  z=torch.randn(7,32)
  from model import rotate_plane
  v=rotate_plane(z,m.ang_g,8)
  assert torch.allclose(z.square().sum(-1)[:,None],v.square().sum(-1),atol=5e-5)

def test_native_gain_and_mirror_same_trainable_count():
  cfg=Config(p=17,h=32,q=8)
  assert sum(p.numel() for p in MirrorSwiGLU(cfg,'gain').parameters())==sum(p.numel() for p in MirrorSwiGLU(cfg,'mirror').parameters())
  assert sum(p.numel() for p in MirrorSwiGLU(cfg,'pair_mirror').parameters())==sum(p.numel() for p in MirrorSwiGLU(cfg,'mirror').parameters())
