import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from run_screen import H, ViewModel, load_data, make_dlinear, tensors

def test_pinned_dlinear_shape_and_series_decomposition():
    model=make_dlinear(252)
    x=torch.randn(3,252,8)
    assert model(x).shape==(3,H,8)

def test_zero_phase_mirror_is_identical_to_fixed_periodic_view():
    torch.manual_seed(1056)
    base1=make_dlinear(252); base2=make_dlinear(252); base2.load_state_dict(base1.state_dict())
    fixed=ViewModel(base1,'fixed'); mirror=ViewModel(base2,'mirror')
    x=torch.randn(4,252,8)
    assert torch.equal(fixed(x),mirror(x))

def test_windows_obey_registered_target_bounds_and_channel_scales():
    data,_=load_data(include_audit=False)
    assert len(data['z'])==int(7588*.8) and len(data['scale'])==8
    assert 'audit' not in data
    for split in ('train','dev'):
        x,y,starts=tensors(data,split,252)
        assert len(x)==len(y)>0
        assert np.all(starts+H<=data[split]['hi'])
        if split!='train': assert np.all(starts>=data[split]['lo'])
