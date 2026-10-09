import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma320',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_expert_bank_and_balanced_dispatch():
 a=m.make(32000,0,'aligned');assert a[0].shape==(8,16,16) and a[1].shape==(128,8)
 assert a[5].shape==(4096,16) and a[6].shape==(4096,16)
 assert torch_all_counts_equal(a[7])
def torch_all_counts_equal(x):
 import torch
 return torch.unique(torch.bincount(x,minlength=128)).numel()==1
def test_givens_norm():
 import torch
 z=torch.randn(8);assert abs(float(m.rotate(z,torch.tensor(.4)).norm()-z.norm()))<1e-5
