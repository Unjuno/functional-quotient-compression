import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma299',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_late_tasks_are_novel_and_deterministic():
 a=m.make(29900,0);b=m.make(29900,0)
 assert a[0].shape==(10,256) and (a[0]==b[0]).all()
def test_payload_includes_more_paid_tensor_bytes():
 import torch
 assert len(m.pack([torch.zeros(2),torch.zeros(3)],'m'))>len(m.pack([torch.zeros(2)],'m'))
