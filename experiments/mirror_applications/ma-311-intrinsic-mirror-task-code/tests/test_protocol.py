import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma311',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_teacher_strata_shapes():
 a=m.make(31100,0,'aligned');b=m.make(31100,0,'independent')
 assert a[0].shape==(128,16) and a[1].shape==(8,128,16)
 assert a[3].shape==(8,256,16) and a[4].shape==(8,256) and b[5].shape==(8,16)
def test_givens_rotation_preserves_norm():
 import torch
 z=torch.randn(16);assert abs(float(m.rotate(z,torch.tensor(.7)).norm()-z.norm()))<1e-5
