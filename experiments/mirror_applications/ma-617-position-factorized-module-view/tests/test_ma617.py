import importlib.util
from pathlib import Path
import torch
p=Path(__file__).parents[1]/'source/run_experiment.py';s=importlib.util.spec_from_file_location('ma617',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_route_orders_include_noncommuting_orderings():assert (0,1,2) in m.SEQ and (2,1,0) in m.SEQ and m.SEQ.index((0,1,2))!=m.SEQ.index((2,1,0))
def test_direct_code_and_mirror_share_view_algebra():
 w=torch.tensor([[1.,2.],[-1.,.4]]);a=torch.tensor(.31);assert torch.allclose(m.conj(w,a),m.rot(a)@w@m.rot(a).T)
