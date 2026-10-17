import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'source'))
import io,torch,pytest
from core import N,NativeMLP,native_flat,logits,mirror_q,seed_transform,orbit,invert_orbit,CoeffDecoder,state_bytes,native_payload,best_svd_under_byte_cap,from_svd

@pytest.fixture(autouse=True)
def one_thread():torch.set_num_threads(1)

@pytest.mark.parametrize('r,k',[(0,0),(1,0),(1,2),(3,4)])
def test_invertible_mirror(r,k):
    x=torch.randn(N,dtype=torch.float64)
    assert torch.allclose(mirror_q(mirror_q(x,r,k),r,k,True),x,atol=1e-12)

def test_seed_inverse():
    x=torch.randn(N,dtype=torch.float64)
    assert torch.allclose(seed_transform(seed_transform(x),True),x,atol=1e-12)

@pytest.mark.parametrize('k,r',[(1,1),(2,2),(4,4),(8,1)])
def test_exact_full_dimension_inverse(k,r):
    target=torch.randn(N,dtype=torch.float64)*.2
    inv=invert_orbit(target,k,r)
    assert (orbit(seed_transform(inv),k,r)-target).abs().max()<1e-11

def test_k_and_rounds_change_function():
    z=torch.randn(N,dtype=torch.float64)*.3
    assert (orbit(z,1,1)-orbit(z,4,1)).abs().mean()>1e-4
    assert (orbit(z,1,1)-orbit(z,1,4)).abs().mean()>1e-4

def test_autograd_stays_finite_and_nonzero():
    m=CoeffDecoder(list(range(16)),'mirror',2,2)
    v=m.weights();(v*torch.randn_like(v)).sum().backward()
    assert torch.isfinite(m.coords.grad).all()
    assert m.coords.grad.norm()>1e-8

def test_native_equivalent_flat_and_dtype_storage():
    x=torch.randn(6,64,dtype=torch.float64)
    net=NativeMLP()
    w=native_flat(net)
    assert torch.allclose(net(x),logits(w,x),atol=1e-14)
    assert state_bytes(native_payload(w,torch.float16))<state_bytes(native_payload(w,torch.float64))

def test_serialized_address_metadata_paid():
    a=CoeffDecoder([2,3,5,7],'mirror',4,3)
    raw=io.BytesIO();torch.save(a.payload(),raw)
    b=torch.load(io.BytesIO(raw.getvalue()),weights_only=True)
    other=CoeffDecoder(b['state']['indices'].long(),b['kind'],b['k'],b['rounds'])
    other.load_state_dict(b['state'])
    assert torch.equal(a.weights(),other.weights())
    assert len(raw.getvalue())==state_bytes(a.payload())

def test_svd_budget_and_rank_monotonic():
    w=torch.randn(N,dtype=torch.float64)
    cap=state_bytes(CoeffDecoder(list(range(768)),'mirror',4,4).payload())
    p=best_svd_under_byte_cap(w,cap)
    assert p is not None and state_bytes(p)<=cap
    assert from_svd(p).shape==(N,)
    assert p['rank']<=16
