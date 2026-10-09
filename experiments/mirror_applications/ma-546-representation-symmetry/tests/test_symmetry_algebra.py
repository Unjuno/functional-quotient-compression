import sys
from pathlib import Path
import numpy as np
import torch

SOURCE=Path(__file__).resolve().parents[1]/'source'
sys.path.insert(0,str(SOURCE))
from run_experiment import hadamard,make_codes


def test_hadamard_is_orthogonal():
    q=hadamard(16)
    assert np.allclose(q@q.T,np.eye(16),atol=1e-6)


def test_monomial_inverse_compensation_preserves_linear_output():
    rng=np.random.default_rng(10)
    n,d,o=7,16,5
    z=torch.tensor(rng.normal(size=(n,d)),dtype=torch.float32)
    w=torch.tensor(rng.normal(size=(o,d)),dtype=torch.float32)
    b=torch.tensor(rng.normal(size=(o,)),dtype=torch.float32)
    perm=rng.permutation(d);sign=rng.choice([-1.,1.],d);scale=np.exp(rng.uniform(np.log(.5),np.log(2.),d))
    p=torch.tensor(perm);m=torch.tensor(sign*scale,dtype=torch.float32)
    expected=torch.nn.functional.linear(z,w,b)
    actual=torch.nn.functional.linear(z.index_select(-1,p)*m,w.index_select(1,p)/m,b)
    assert torch.max(torch.abs(expected-actual)).item()<2e-5


def test_dense_orthogonal_inverse_compensation_preserves_linear_output():
    rng=np.random.default_rng(11)
    z=torch.tensor(rng.normal(size=(9,16)),dtype=torch.float32)
    w=torch.tensor(rng.normal(size=(6,16)),dtype=torch.float32)
    q=torch.tensor(hadamard(16),dtype=torch.float32)
    expected=torch.nn.functional.linear(z,w)
    actual=torch.nn.functional.linear(z@q.T,w@q.T)
    assert torch.max(torch.abs(expected-actual)).item()<2e-5


def test_seeded_view_codes_are_reproducible():
    a=make_codes(42,32);b=make_codes(42,32)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    assert a[0].shape==(16,32)
    assert all(np.array_equal(np.sort(row),np.arange(32)) for row in a[0])
