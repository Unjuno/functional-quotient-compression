import importlib.util
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma603',ROOT/'source'/'run_ma603.py');ma=importlib.util.module_from_spec(spec);spec.loader.exec_module(ma)
def test_teacher_is_deterministic():
    W,b=ma.world(11);x=ma.data(11,'query',32);y=ma.teacher(x,W,b);assert y.shape==(32,8) and np.array_equal(y,ma.teacher(x,W,b))
def test_controls_forward_and_backprop():
    x=torch.randn(12,8)
    for m in ma.METHODS:
        net=ma.Net(m);y=net(x);assert y.shape==(12,8);y.square().mean().backward();assert any(p.grad is not None for p in net.parameters())
def test_givens_teacher_preserves_norm():
    x=np.random.default_rng(7).normal(size=(32,8)).astype('float32');W,b=ma.world(3);h=x@W.T+b;y=ma.teacher(x,W,b);assert np.allclose(np.sum(h*h,axis=1),np.sum(y*y,axis=1),atol=1e-5)
def test_draw29_uniform_replay():
    import hashlib,json
    d=json.loads((ROOT/'source'/'draw29_exclusions.json').read_text());pool=d['pool_ids'];seed=bytes.fromhex(d['seed_hex']);ph=bytes.fromhex(d['pool_sha256']);ctr=d['rejection_counter'];digest=hashlib.sha256(seed+ph+ctr.to_bytes(4,'big')).digest();n=len(pool);limit=(1<<256)-((1<<256)%n);v=int.from_bytes(digest,'big')
    assert len(pool)==d['eligible_count'] and hashlib.sha256('\n'.join(pool).encode()).hexdigest()==d['pool_sha256'];assert v<limit and v%n==d['selection_index_zero_based'];assert pool[v%n]=='MA-603'
def test_serialized_payload_contains_all_parameters(tmp_path):
    from safetensors.torch import load_file
    net=ma.Net('mirror_givens');p=tmp_path/'model.safetensors';n,sha=ma.payload(net,p);loaded=load_file(str(p))
    expected=sum(v.numel() for v in net.state_dict().values())
    assert list(loaded)==['p'] and loaded['p'].numel()==expected and p.stat().st_size==n and len(sha)==64
