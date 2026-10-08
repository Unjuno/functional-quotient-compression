import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma951',ROOT/'source'/'run_ma951.py');ma=importlib.util.module_from_spec(spec);spec.loader.exec_module(ma)
def test_circulant_equivariance_exact():
    k=np.array([.1,.2,.3,.4],dtype='float32');C=ma.group_matrix(k);x=np.arange(4,dtype='float32')
    for r in range(4):assert np.allclose(np.roll(C@x,r),C@np.roll(x,r))
def test_teacher_shapes_and_structures():
    for v in ma.VARIANTS:
        C,D=ma.teacher(19,v);assert C.shape==(4,4) and D.shape==(4,4,4)
    assert np.allclose(ma.teacher(19,'aligned')[0],ma.teacher(19,'aligned')[0])
def test_models_forward_and_gradients():
    x=torch.randn(12,4);t=torch.randint(0,4,(12,))
    for m in ma.METHODS:
        net=ma.Model(m);y=net(x,t);assert y.shape==(12,4);y.square().mean().backward();assert any(p.grad is not None for p in net.parameters())
def test_draw31_replay():
    d=json.loads((ROOT/'source'/'draw31_exclusions.json').read_text());pool=d['pool_ids'];ph=bytes.fromhex(d['pool_sha256']);seed=bytes.fromhex(d['seed_hex']);dig=hashlib.sha256(seed+ph+d['rejection_counter'].to_bytes(4,'big')).digest();n=len(pool);v=int.from_bytes(dig,'big');limit=(1<<256)-((1<<256)%n)
    assert len(pool)==d['eligible_count'] and hashlib.sha256('\n'.join(pool).encode()).hexdigest()==d['pool_sha256'];assert v<limit and v%n==d['selection_index_zero_based'];assert pool[v%n]=='MA-951'
