import importlib.util
from pathlib import Path
import torch
p=Path(__file__).parents[1]/'source/run_experiment.py';s=importlib.util.spec_from_file_location('ma616',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_identity_conjugation():assert torch.allclose(m.conj(torch.eye(2),torch.tensor(.7)),torch.eye(2),atol=1e-6)
def test_direct_view_code():
 w=torch.tensor([[1.,2.],[-.5,.3]]);a=torch.tensor(.42);q=m.rot(a);assert torch.allclose(m.conj(w,a),q@w@q.T)
