import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from engine import teacher
from model import ProductExperts,METHODS

def test_product_pair_mirror_exactly_represents_aligned_teacher():
    seed=41;t=teacher('aligned',seed);m=ProductExperts('mirror',3)
    g=torch.Generator().manual_seed(seed)
    base=torch.randn(2,8,8,generator=g)*.10+torch.eye(8).unsqueeze(0)*.12
    ai=torch.rand(2,2,generator=g)*1.2-.6;ao=torch.rand(2,2,generator=g)*1.2-.6
    with torch.no_grad():
        m.weight.copy_(base);m.input_angle.copy_(ai);m.output_angle.copy_(ao)
    assert torch.allclose(m.factors(),t,atol=1e-7)

def test_all_four_factor_pairs_are_distinct_and_shape_is_valid():
    m=ProductExperts('mirror',21);roles=torch.arange(4);x=torch.randn(4,8)
    assert m(x,roles).shape==(4,8)
    assert m.factors().shape==(2,4,8,8)

def test_all_registered_controls_have_gradients():
    x=torch.randn(8,8);r=torch.arange(8)%4
    for method in METHODS:
        m=ProductExperts(method,99);m(x,r).square().mean().backward()
        assert all(p.grad is not None for p in m.parameters())
