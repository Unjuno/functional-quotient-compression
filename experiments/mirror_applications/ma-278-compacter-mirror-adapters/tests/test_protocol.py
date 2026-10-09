import importlib.util
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[1]/'source'/'run.py';sp=importlib.util.spec_from_file_location('ma278',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

def test_kronecker_atoms_shape():
 g=torch.Generator().manual_seed(4);a,s,f=m.atoms(g);assert a.shape==(m.A,m.D,m.D)

def test_task_shapes():
 w,x,xv,b,s,f,aligned,ind=m.make(27800,0);assert aligned.shape==ind.shape==(m.T,m.D,m.D)

def test_payload_charges_codes():
 w=torch.randn(m.D,m.D);b=torch.randn(m.A,m.D,m.D);assert len(m.pack([w,b,torch.zeros(1)],'x',[list(w.shape),list(b.shape),[1]]))<len(m.pack([w,b,torch.zeros(20)],'x',[list(w.shape),list(b.shape),[20]]))
