import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma330',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_cache_shapes_and_roles():
 a=m.make(33000,0,'aligned')
 assert a[0].shape==(16,32,16) and a[1].shape==(16,32,16) and a[2].shape==(16,8,16)
def test_rotation_preserves_token_vector_norm():
 import torch
 x=torch.randn(32,16);assert torch.allclose(m.rotate(x,torch.tensor(.8)).norm(),x.norm(),atol=1e-5)
