import io
import torch
from source.model import N, NativeMLP, weight_to_flat,mirror_q,mirror_increment,nonlinear_orbit,seed_transform
from source.dev_direction_search import AddressedMirrorMLP
from source.replicate_direction import data_world

torch.set_num_threads(1)

def test_direction_indices_are_paid():
    m=256
    sparse=AddressedMirrorMLP(torch.arange(m),True)
    assert [k for k in sparse.state_dict()] == ['coords','indices']
    assert sum(p.numel() for p in sparse.parameters())==m
    assert sparse.indices.dtype==torch.int16
    b=io.BytesIO();torch.save({'m':m,'state':sparse.state_dict()},b)
    assert len(b.getvalue())>=m*8 +m*2

def test_sparse_mirror_identity_when_full_set():
    t=NativeMLP()
    full=AddressedMirrorMLP(torch.arange(N),True)
    from source.model import invert_nonlinear_orbit
    with torch.no_grad():full.coords.copy_(invert_nonlinear_orbit(weight_to_flat(t)))
    assert (full.weights()-weight_to_flat(t)).abs().max()<1e-9

def test_noncommuting_parallel_mirror_directions():
    torch.manual_seed(2)
    z=torch.randn(N,dtype=torch.float64)
    a=mirror_q(mirror_q(z,0,0),1,1)
    b=mirror_q(mirror_q(z,1,1),0,0)
    assert torch.norm(a-b).item()>1e-2

def test_all_dataset_splits_disjoint_and_stratified():
    for world in (618310,618311,618312):
        _,_,p=data_world(world)
        assert all(len(p[q])==len(set(p[q])) for q in p)
        assert not (set(p['train']) & set(p['test']))
        assert not (set(p['dev']) & set(p['test']))
        assert not (set(p['dev']) & set(p['train']))
        assert len(p['test'])==360

def test_analytic_numerical_gradient_wrt_coordinates():
    torch.manual_seed(410)
    model=AddressedMirrorMLP(torch.arange(16),True)
    model.coords.data.copy_(torch.randn(16,dtype=torch.float64)*.03)
    weights=model.weights()
    loss=(weights.square()).mean()
    loss.backward()
    h=1e-5
    with torch.no_grad():
        v=model.coords[2].item()
        model.coords[2]=v+h
        p=float(model.weights().square().mean())
        model.coords[2]=v-h
        n=float(model.weights().square().mean())
        model.coords[2]=v
    rel=abs((p-n)/(2*h)-float(model.coords.grad[2]))
    assert rel<1e-7,rel
