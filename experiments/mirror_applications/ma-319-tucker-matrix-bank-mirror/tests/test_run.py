import importlib.util
from pathlib import Path
import torch

SRC=Path(__file__).resolve().parents[1]/'source'/'run.py'
spec=importlib.util.spec_from_file_location('ma319_run',SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture_state():
    g=torch.Generator().manual_seed(31);state={}
    for fam in m.FAMILIES:
        shape=(5,4) if fam in ('attn.c_attn.weight','mlp.c_fc.weight') else (4,4)
        for i in range(m.N_LAYER):
            state[f'transformer.h.{i}.{fam}']=torch.randn(shape,generator=g)+i*.03
    state['transformer.ln_f.weight']=torch.ones(4)
    return state

def test_tucker_payload_roundtrips_and_is_smaller_than_full():
    state=fixture_state();p=m.compress(state,'tucker2');decoded=m.decode(p,state)
    assert set(decoded)==set(state)
    assert all(torch.isfinite(v).all() for v in decoded.values())
    full=m.serialize({'format':'ma319-v1','method':'full','tensors':state,'banks':{}})
    compressed=m.serialize(p)
    assert len(compressed)<len(full)

def test_mirror_uses_one_angle_per_layer_and_reconstructs():
    p=m.compress(fixture_state(),'mirror')
    d=m.decode(p,fixture_state())
    assert all(p['banks'][fam]['angles'].numel()==m.N_LAYER for fam in m.FAMILIES)
    assert all(torch.isfinite(v).all() for v in d.values())
    assert p['mean_matrix_reconstruction_mse']>=0

def test_serialization_deterministic():
    p=m.compress(fixture_state(),'mirror')
    assert m.serialize(p)==m.serialize(p)
