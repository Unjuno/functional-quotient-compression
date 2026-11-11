from types import SimpleNamespace
import numpy as np
import importlib.util
from pathlib import Path
mod_path=Path(__file__).parents[1]/'source'/'experiment.py'
spec=importlib.util.spec_from_file_location('ma877_experiment',mod_path)
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
rope_transform=mod.rope_transform


def test_gpt_neox_partial_rope_inverse_roundtrip():
    cfg = SimpleNamespace(model_type='gpt_neox', hidden_size=64, num_attention_heads=1, rotary_pct=.25, rope_theta=10000)
    x = np.random.default_rng(17).normal(size=(37,64))
    assert np.max(np.abs(rope_transform(rope_transform(x,cfg),cfg,inverse=True)-x)) < 1e-12


def test_llama_rope_inverse_roundtrip_and_theta_change():
    a = SimpleNamespace(model_type='llama', hidden_size=64, num_attention_heads=1, rope_theta=10000)
    b = SimpleNamespace(model_type='llama', hidden_size=64, num_attention_heads=1, rope_theta=100000)
    x = np.random.default_rng(23).normal(size=(51,64))
    encoded=rope_transform(x,a)
    assert np.max(np.abs(rope_transform(encoded,a,inverse=True)-x)) < 1e-12
    assert np.max(np.abs(rope_transform(x,a)-rope_transform(x,b))) > 1e-3


def test_target_rope_reapplied_to_position_free_content():
    source = SimpleNamespace(model_type='gpt_neox', hidden_size=64, num_attention_heads=1, rotary_pct=.25, rope_theta=10000)
    target = SimpleNamespace(model_type='llama', hidden_size=64, num_attention_heads=1, rope_theta=100000)
    x=np.random.default_rng(31).normal(size=(19,64))
    source_key=rope_transform(x,source)
    content=rope_transform(source_key,source,inverse=True)
    target_key=rope_transform(content,target)
    assert np.max(np.abs(rope_transform(target_key,target,inverse=True)-x)) < 1e-12


def test_serialized_payload_sizes_are_actual_and_deterministic():
    rng=np.random.default_rng(41)
    maps=rng.normal(size=(6,3,4))
    flat=maps.reshape(6,-1)
    mean=flat.mean(axis=0)
    _,_,basis=np.linalg.svd(flat-mean,full_matrices=False)
    refs=[(i,0,i%2,i,0) for i in range(6)]
    item={'pair':('src','tgt'),'maps':maps,'refs':refs}
    full,shared=mod.serialize_family_payloads([item],mean,basis,2)
    assert isinstance(full,bytes) and isinstance(shared,bytes)
    assert len(full)>len(shared)>0
    assert mod.serialize_family([item],mean,basis,2)==(len(full),len(shared))
    assert full==mod.serialize_family_payloads([item],mean,basis,2)[0]
    assert shared==mod.serialize_family_payloads([item],mean,basis,2)[1]
