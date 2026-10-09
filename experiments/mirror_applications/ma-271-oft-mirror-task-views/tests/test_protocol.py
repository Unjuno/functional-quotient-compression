import importlib.util
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[1]/'source'/'run.py';spec=importlib.util.spec_from_file_location('ma271',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_plane_is_orthogonal_and_preserves_geometry():
 q=m.plane(torch.tensor(.37));assert torch.allclose(q.T@q,torch.eye(m.DOUT),atol=1e-6)
 x=torch.randn(12,m.DOUT);assert torch.allclose(torch.cdist(x@q,x@q),torch.cdist(x,x),atol=1e-5)

def test_angle_fit_recovers_orbit():
 x=torch.randn(128,m.DOUT);theta=torch.tensor(.52);a,_=m.angle_fit(x,x@m.plane(theta));assert abs(a-float(theta))<1e-4

def test_oft_cayley_orthogonality():
 x=torch.randn(64,m.DOUT);q,_,_=m.orthogonal_fit(x,x@m.plane(torch.tensor(.23)),steps=4);assert torch.allclose(q.T@q,torch.eye(m.DOUT),atol=1e-5)
