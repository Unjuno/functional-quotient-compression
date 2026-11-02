import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma527_run',ROOT/'source'/'run_givens.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def test_zero_angles_identity():
    x=np.arange(16,dtype=np.float32)
    assert np.array_equal(runner.givens(x,np.zeros(8,dtype=np.float32)),x)

def test_givens_preserves_norm():
    rng=np.random.default_rng(42);x=rng.normal(size=(7,16));a=rng.normal(size=(7,8))
    y=runner.givens(x,a)
    assert np.allclose(np.linalg.norm(x,axis=1),np.linalg.norm(y,axis=1),atol=1e-12)

def test_transform_is_pair_local():
    x=np.zeros(16);x[0]=1
    y=runner.givens(x,np.full(8,np.pi/2))
    assert np.allclose(y[0:2],[0,1],atol=1e-7)
    assert np.count_nonzero(y[2:])==0

def test_numpy_torch_agree():
    import torch
    x=np.arange(16,dtype=np.float32);a=np.linspace(-1,1,8,dtype=np.float32)
    n=runner.givens(x,a);t=runner.givens(torch.tensor(x),torch.tensor(a)).numpy()
    assert np.allclose(n,t,atol=1e-7)
