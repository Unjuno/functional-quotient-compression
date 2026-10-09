import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma325',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_tt_embedding_and_domains():
 a=m.make(32500,0,'aligned')
 assert a[1].shape==(32,2,4,4,2) and a[3].shape==(32,256,256) and a[4].shape==(32,64)
def test_core_rotation_preserves_frobenius_norm():
 import torch
 c=torch.randn(2,4,4,2);assert abs(float(m.mod_core(c,torch.tensor(.7)).norm()-c.norm()))<1e-5
