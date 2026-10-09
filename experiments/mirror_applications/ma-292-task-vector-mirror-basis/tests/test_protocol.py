import importlib.util
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[1]/'source'/'run.py';sp=importlib.util.spec_from_file_location('ma292',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

def test_task_data_shapes_and_functions():
 b,c,d,x,y,xq,yq,tr,held,pairs=m.make(29200,0);assert x.shape==(m.N,64,m.D) and y.shape==(m.N,64);assert torch.allclose(y[2],x[2]@d[2].unsqueeze(-1).squeeze(-1))

def test_heldout_tasks_disjoint():
 *_,tr,held,pairs=m.make(29200,0);assert set(tr).isdisjoint(held)

def test_payload_charges_basis():
 a=torch.randn(m.K,m.D);assert len(m.pack([a],'x'))<len(m.pack([a,torch.randn(2,m.D)],'x'))
