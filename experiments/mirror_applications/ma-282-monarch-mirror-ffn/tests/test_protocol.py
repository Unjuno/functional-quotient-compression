import importlib.util
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[1]/'source'/'run.py';sp=importlib.util.spec_from_file_location('ma282',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

def test_monarch_shape():
 a=torch.randn(m.D//m.B,m.B,m.B);assert m.monarch(a,a).shape==(m.D,m.D)

def test_task_shapes():
 w,x,xv,a,b,d,aligned,ind=m.make(28200,0);assert aligned.shape==ind.shape==(m.T,m.D,m.D)

def test_payload_charges_all_factors():
 a=torch.randn(2,4,4);x=m.pack([a],'x');y=m.pack([a,torch.randn(1)],'x');assert len(y)>len(x)
