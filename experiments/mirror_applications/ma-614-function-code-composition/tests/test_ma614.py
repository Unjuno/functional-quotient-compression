import importlib.util
from pathlib import Path
import torch
p=Path(__file__).parents[1]/'source/run_experiment.py';spec=importlib.util.spec_from_file_location('ma614',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_composition_function_is_bounded():
 x=torch.linspace(-3.14,3.14,100);y=m.target(x,torch.tensor(.3),torch.tensor(-.7));assert torch.isfinite(y).all() and y.abs().max()<=1

def test_serialized_payload_reproducible():
 a=m.pack('mirror',1,torch.tensor([[1,2]]),torch.tensor([.1,.2]),torch.zeros(1,10));b=m.pack('mirror',1,torch.tensor([[1,2]]),torch.tensor([.1,.2]),torch.zeros(1,10));assert a==b
