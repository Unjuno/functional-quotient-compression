import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma327',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_observed_and_heldout_cartesian_cells():
 a=m.make(32700,0,'aligned');obs,held=a[7],a[8]
 assert obs.shape==(16,16) and int(held.sum())>0 and obs[:,0].all()
 assert bool((obs|held).all()) and not bool((obs&held).any())
def test_orthonormal_matrix_bank():
 import torch
 bank=m.make(32700,0,'aligned')[0].flatten(1)
 assert torch.allclose(bank@bank.T,torch.eye(8),atol=1e-5)
