import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma319',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_bank_and_layer_shapes():
 a=m.make(31900,0,'aligned')
 assert a[0].shape==(8,32,32) and a[1].shape==(64,8) and a[5].shape==(64,128,32)
def test_givens_preserves_coefficient_norm():
 import torch
 z=torch.randn(8);assert abs(float(m.rotate(z,torch.tensor(.8)).norm()-z.norm()))<1e-5
