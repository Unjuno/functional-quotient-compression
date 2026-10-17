import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from numba import set_num_threads

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'source'))
# The original implementation is the sibling inference_microaudit/source directory.
import os
legacy_source=HERE.parent/'source'
if not legacy_source.exists():legacy_source=Path(os.environ['MA1164_MICROAUDIT_SOURCE'])
sys.path.insert(0,str(legacy_source))
from fused_online import forward
from online_operator import signs, hadamard, materialize, dense_forward, direct_nonlinear_sandwich

set_num_threads(1)
torch.set_num_threads(1)

def fixtures(d,b,k,r):
    rng=np.random.default_rng(44+d+k+r)
    x=rng.standard_normal((b,d))*0.5
    u=rng.standard_normal(d)*0.03
    codes=np.stack([np.stack([signs(i,j,d).to(torch.int8).numpy() for j in range(k)]) for i in range(r)])
    return x,u,codes

@pytest.mark.parametrize('d,b,k,r',[(16,1,1,1),(64,3,4,3),(128,7,2,4),(256,2,8,2)])
@pytest.mark.parametrize('parallel',[False,True])
def test_linear_matched_to_preexpanded_dense(d,b,k,r,parallel):
    x,u,codes=fixtures(d,b,k,r)
    h=hadamard(torch.eye(d,dtype=torch.float64))
    mats=torch.stack([materialize(torch.from_numpy(u),torch.from_numpy(codes[i,j]).double(),h) for i in range(r) for j in range(k)])
    ref=dense_forward(torch.from_numpy(x),mats,k,r).numpy()
    got=forward(x,u,codes,parallel=parallel)
    assert np.max(np.abs(got-ref)) < 2e-12

@pytest.mark.parametrize('parallel',[False,True])
def test_nonlinear_inside_mirror_matched_to_pytorch(parallel):
    d,b,k,r=64,4,3,3
    x,u,codes=fixtures(d,b,k,r)
    z=torch.from_numpy(x).clone();uu=torch.from_numpy(u)
    for t in range(r):
        v=torch.stack([direct_nonlinear_sandwich(z,uu,torch.from_numpy(codes[t,i]).double()) for i in range(k)]).mean(0)
        z=z+0.2*v
    got=forward(x,u,codes,nonlinear_inside=True,parallel=parallel)
    assert np.max(np.abs(got-z.numpy())) < 3e-13

def test_fixed_codes_reproduce():
    x,u,codes=fixtures(32,5,2,2)
    a=forward(x,u,codes);b=forward(x,u,codes)
    assert np.array_equal(a,b)

def test_no_temporary_dense_matrix_inputs_and_validation():
    x,u,codes=fixtures(64,2,2,2)
    assert codes.nbytes==64*2*2
    assert u.nbytes == 64*8
    with pytest.raises(ValueError):forward(x,u,codes[:,:,:-1])
    with pytest.raises(ValueError):forward(x[:,:-1],u,codes)
    bad=codes.copy();bad[0,0,0]=0
    with pytest.raises(ValueError):forward(x,u,bad)
