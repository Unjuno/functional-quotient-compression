import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from run_screen import CTX,H,RoleModel,load_data,make_model

def test_official_forward_parity_at_zero_address():
    base=make_model(1055); x=torch.randn(3,CTX)
    expected=base(x,torch.ones_like(x))
    actual=RoleModel(base,'shared')(x,torch.zeros(3,dtype=torch.long),torch.zeros(3,dtype=torch.long))
    assert torch.equal(expected,actual)

def test_phase_pair_rotation_preserves_pair_norm():
    pair=torch.tensor([[1.25,-.75]])
    angle=torch.tensor([[.4]])
    c,s=pair[:,0:1],pair[:,1:2]
    cr,sr=c*torch.cos(angle)-s*torch.sin(angle),c*torch.sin(angle)+s*torch.cos(angle)
    assert torch.allclose(c*c+s*s,cr*cr+sr*sr,atol=1e-6)

def test_targets_respect_frozen_split_boundaries():
    data,_=load_data()
    for (_,split),d in data.items():
        assert np.all(d['starts']>=d['lo']+CTX)
        assert np.all(d['starts']+H<=d['hi'])
