import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from engine import make_teacher,target
from model import ExpertBank,METHODS

def test_aligned_rank1_mirror_represents_teacher():
    t=make_teacher('aligned',75);m=ExpertBank('mirror_rank1',9)
    with torch.no_grad():
        m.win.copy_(t['wi']);m.wout.copy_(t['wo']);m.input_angle.copy_(t['ai']);m.output_angle.copy_(t['ao']);m.left.copy_(t['left']);m.right.copy_(t['right'])
    g=torch.Generator().manual_seed(7);x=torch.randn(128,16,generator=g);r=torch.arange(128)%4
    assert torch.allclose(m(x,r),target(t,x,r),atol=1e-7)

def test_private_rank_changes_parameterization():
    models={k:ExpertBank(k,11) for k in METHODS}
    assert sum(p.numel() for p in models['mirror_rank2'].parameters())>sum(p.numel() for p in models['mirror_rank1'].parameters())
    x=torch.randn(8,16);r=torch.arange(8)%4
    for m in models.values():
        m(x,r).square().mean().backward()
        assert all(p.grad is not None for p in m.parameters())

def test_independent_control_has_one_full_mlp_per_role():
    m=ExpertBank('independent',15)
    assert m.win.shape==(4,32,16) and m.wout.shape==(4,16,32)
