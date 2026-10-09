import importlib.util
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[1]/'source'/'run.py';sp=importlib.util.spec_from_file_location('ma273',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

def test_butterfly_product_orthogonal():
 g=torch.Generator().manual_seed(1);code=[torch.randn(len(p),generator=g) for p in m.SCHEDULE];q=m.boft_matrix(code);assert torch.allclose(q.T@q,torch.eye(m.D),atol=2e-6)

def test_orbit_reconstruction():
 w,x,a,b,aligned,ind=m.make(27300,0);assert torch.allclose(aligned[3].T@aligned[3],torch.eye(m.D),atol=2e-6)

def test_payload_code_charged():
 w=torch.randn(m.D,m.D);assert len(m.serialize(w,'s',torch.zeros(1)))<len(m.serialize(w,'s',torch.zeros(100)))
