import importlib.util
from pathlib import Path
import numpy as np
import torch

P=Path(__file__).parents[1]/'source'/'experiment.py'
spec=importlib.util.spec_from_file_location('ma881',P); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_straight_through_top1_is_hard_forward_and_differentiable():
    logits=torch.tensor([[0.1,2.0],[3.0,0.0]],requires_grad=True)
    gate=m.straight_through_top1(logits)
    assert torch.equal(gate.detach(),torch.tensor([[0.,1.],[1.,0.]]))
    (gate*torch.tensor([[1.,2.],[3.,4.]])).sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

def test_rope_inverse_roundtrip():
    class C: model_type='llama'; hidden_size=8; num_attention_heads=1; rope_theta=10000.
    x=np.random.default_rng(1).normal(size=(3,5,8))
    y=m.rope_transform(m.rope_transform(x,C()),C(),inverse=True)
    assert np.allclose(x,y,atol=1e-10)

def test_router_output_shape():
    router=m.TokenRouter()
    assert router(torch.zeros(7,m.D+3)).shape==(7,2)
