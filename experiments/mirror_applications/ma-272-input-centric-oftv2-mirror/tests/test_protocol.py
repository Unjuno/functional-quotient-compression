import importlib.util
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[1]/'source'/'run.py';spec=importlib.util.spec_from_file_location('ma272',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_input_and_materialized_forms_equal():
 w=torch.randn(m.D,m.D);x=torch.randn(31,m.D);q=m.givens(torch.tensor(.42));assert m.relerr(m.apply_input(x,w,q),x@m.materialize(w,q))<1e-6

def test_independent_orthogonal_input_form_equal():
 w,x,a,angs,mats=m.make(27200,0);q=mats[2];assert torch.allclose(m.apply_input(a,w,q),a@m.materialize(w,q),atol=1e-6)

def test_view_payload_charged():
 w=torch.randn(m.D,m.D);assert len(m.payload(w,'m',torch.zeros(1)))<len(m.payload(w,'m',torch.zeros(m.D*m.D)))
