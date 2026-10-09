import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma307',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_task_stream_shape_and_reproducibility():
 a=m.make(30700,0);b=m.make(30700,0)
 assert a[0].shape==(12,2048) and a[1].shape==(128,)
 assert (a[0]==b[0]).all()
def test_sparse_residual_serializer_charges_indices_and_values():
 import torch
 z=torch.zeros(2,8);z[1,3]=1
 idx,val=m.sparse(z,.02)
 assert idx.shape==(1,2) and val.shape==(1,)
 assert idx.dtype==torch.int16 and val.dtype==torch.float16
