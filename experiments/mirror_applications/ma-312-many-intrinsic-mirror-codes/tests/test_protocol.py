import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma312',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_many_task_generator_shapes():
 a=m.make(31200,0,'aligned');b=m.make(31200,0,'independent')
 assert a[0].shape==(256,32) and a[1].shape==(64,64,32) and a[3].shape==(64,128,32)
 assert a[4].shape==(64,128) and b[5].shape==(64,32)
def test_rotation_batch_norm():
 import torch
 z=torch.randn(32);theta=torch.linspace(0,1,64);out=m.rotate_batch(z,theta)
 assert torch.allclose(out.norm(dim=1),z.norm().expand(64),atol=1e-5)
