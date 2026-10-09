import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma322',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_tt_matrix_shape():
 import torch
 g=(torch.randn(1,2,2,2),torch.randn(2,2,2,2),torch.randn(2,2,2,2),torch.randn(2,2,2,1))
 assert m.tt_matrix(g).shape==(16,16)
def test_teacher_strata_shapes():
 a=m.make(32200,0,'aligned');b=m.make(32200,0,'independent')
 assert a[0][1].shape==(2,2,2,2) and a[1].shape==(64,2,2,2,2) and a[3].shape==(64,64,16)
