import importlib.util
from pathlib import Path

import torch

SRC=Path(__file__).resolve().parents[1]/'source'/'run_experiment.py'
spec=importlib.util.spec_from_file_location('ma385',SRC);ma=importlib.util.module_from_spec(spec);spec.loader.exec_module(ma)


def test_rotation_preserves_norm_and_zero_is_identity():
    x=torch.randn(32); assert torch.allclose(ma.rotate(x,torch.tensor(0.)),x)
    assert torch.allclose(ma.rotate(x,torch.tensor(.61)).norm(),x.norm(),atol=1e-6)


def test_shared_aligned_task_residuals_follow_rotation_orbit():
    _,_,_,_,deltas=ma.make_world(38501,'aligned')
    assert torch.allclose(deltas.norm(dim=1),deltas[0].norm().expand(8),atol=1e-5)


def test_task_retrieval_uses_only_input_and_keys():
    data,keys,_,_,_=ma.make_world(38501,'aligned')
    slots=torch.cat([ma.nearest(x,keys) for x in data['test'][0]])
    target=torch.arange(8).repeat_interleave(512)
    assert (slots==target).float().mean()>=.95


def test_all_registered_controls_have_task_prompt_tables():
    for method in ('explicit','tied','scalar','mirror','private_rank4','hyper'):
        model=ma.ExpertPrompts(method)
        assert model.expert_table().shape==(8,32)

