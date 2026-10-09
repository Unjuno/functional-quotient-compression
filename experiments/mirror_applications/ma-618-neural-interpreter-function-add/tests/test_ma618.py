import importlib.util
from pathlib import Path
import torch
p=Path(__file__).parents[1]/'source/run_experiment.py';s=importlib.util.spec_from_file_location('ma618',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_grid_and_support_protocol():assert m.GRID==1440 and m.NS==16 and m.NQ==512
def test_phase_and_coeff_fit_are_finite():
 x=torch.linspace(-2,2,16);y=torch.sin(2*x+.3);a=m.fit_mirror(m.NFREQ,x,y);b=m.fit_coeff(m.NFREQ,x,y);assert torch.isfinite(a[2]).all() and torch.isfinite(b[1]).all()
