import importlib.util
from pathlib import Path
import torch
SRC=Path(__file__).resolve().parents[1]/'source'/'run.py';spec=importlib.util.spec_from_file_location('ma333_run',SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_relu_positive_rescale_preserves_function():
 p,x,_=m.world(33301,'relu');s=torch.exp(torch.linspace(-.6,.6,m.HIDDEN));a=m.forward(x,p,'relu');b=m.forward(x,m.transform(p,'relu_scale',s),'relu');assert torch.allclose(a,b,atol=1e-5,rtol=1e-5)

def test_tanh_sign_flip_preserves_function():
 p,x,_=m.world(33302,'tanh');s=torch.where(torch.arange(m.HIDDEN)%2==0,1.,-1.);a=m.forward(x,p,'tanh');b=m.forward(x,m.transform(p,'tanh_sign',s),'tanh');assert torch.allclose(a,b,atol=1e-5,rtol=1e-5)

def test_uncoupled_controls_change_function():
 p,x,_=m.world(33303,'relu');s=torch.exp(torch.linspace(-.6,.6,m.HIDDEN));assert not torch.allclose(m.forward(x,p,'relu'),m.forward(x,m.transform(p,'relu_uncoupled',s),'relu'))

def test_serialized_weight_arrays_roundtrip(tmp_path):
 p,_,_=m.world(33304,'tanh');a=m.weight_arrays(p);n,h=m.pack(a,tmp_path/'a.zip');b=m.load(tmp_path/'a.zip');assert n==(tmp_path/'a.zip').stat().st_size and len(h)==64
