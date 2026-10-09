import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma314',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_task_rank_strata_shapes():
 a=m.make(31400,0,'aligned');b=m.make(31400,0,'independent')
 assert a[0].shape==(128,32) and a[1].tolist()==[4,8,16,32]*6
 assert a[2].shape==(24,96,32) and a[4].shape==(24,128,32)
 assert b[5].shape==(24,128)
def test_rotation_norm_and_truncation():
 import torch
 z=torch.randn(32);q=m.quarter(z,8);assert torch.count_nonzero(q[8:])==0
 assert abs(float(m.rotate(z,torch.tensor(.5)).norm()-z.norm()))<1e-5
