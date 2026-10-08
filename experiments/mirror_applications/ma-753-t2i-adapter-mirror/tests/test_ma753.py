import importlib.util
import hashlib,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma753',ROOT/'source'/'run_ma753.py');ma=importlib.util.module_from_spec(spec);spec.loader.exec_module(ma)
def test_teacher_shapes_and_determinism():
    for v in ma.VARIANTS:
        a=ma.teacher(11,v);b=ma.teacher(11,v);assert a.shape==(4,8,8) and np.array_equal(a,b)
def test_all_methods_forward_and_backprop():
    x=torch.randn(16,8);c=torch.randint(0,4,(16,))
    for method in ma.METHODS:
        model=ma.Adapter(method);y=model(x,c);assert y.shape==(16,8);y.square().mean().backward();assert any(p.grad is not None for p in model.parameters())
def test_aligned_teacher_is_givens_orbit():
    mats=ma.teacher(7,'aligned');s=np.linalg.svd(mats,compute_uv=False)
    for i in range(1,4):assert np.allclose(s[0],s[i],atol=1e-5)
def test_draw30_replay():
    d=json.loads((ROOT/'source'/'draw30_exclusions.json').read_text());pool=d['pool_ids'];seed=bytes.fromhex(d['seed_hex']);ph=bytes.fromhex(d['pool_sha256']);digest=hashlib.sha256(seed+ph+d['rejection_counter'].to_bytes(4,'big')).digest();n=len(pool);v=int.from_bytes(digest,'big');limit=(1<<256)-((1<<256)%n)
    assert len(pool)==d['eligible_count'] and hashlib.sha256('\n'.join(pool).encode()).hexdigest()==d['pool_sha256'];assert v<limit and v%n==d['selection_index_zero_based'];assert pool[v%n]=='MA-753'
def test_actual_adapter_payload_roundtrip(tmp_path):
    from safetensors.torch import load_file
    model=ma.Adapter('mirror_givens');path=tmp_path/'adapter.safetensors';n,sha=ma.payload(model,path);loaded=load_file(str(path))
    assert loaded['p'].numel()==sum(p.numel() for p in model.state_dict().values());assert path.stat().st_size==n and len(sha)==64
