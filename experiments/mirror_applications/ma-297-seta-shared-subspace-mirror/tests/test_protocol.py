import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma297',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_task_structure_and_shared_support():
 ds,shared,priv,x,y=m.make(29700,0)
 assert ds.shape==(8,128) and x.shape==(8,640,128)
 assert (ds!=0).sum(1).tolist()==[20]*8
 assert set(shared.tolist()).isdisjoint(set(priv.flatten().tolist()))
def test_actual_serialization_charges_each_tensor():
 import torch
 a=m.pack([torch.zeros(2)],'a');b=m.pack([torch.zeros(2),torch.zeros(3)],'b')
 assert len(b)>len(a)
