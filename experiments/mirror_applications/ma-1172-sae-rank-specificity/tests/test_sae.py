import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import torch
import torch.nn.functional as F
from run import SAE, METHODS, WORLDS, dataset, teacher, sparse_relu_topk, fixed_maps,spectrum

def test_shapes_sparsity_gradients():
    torch.set_num_threads(1)
    for method in METHODS:
        torch.manual_seed(5)
        m=SAE(method)
        X=torch.randn(20,32)
        z,y=m.encode_reconstruct(X)
        assert z.shape==(20,64) and y.shape==(20,32)
        assert (z>1e-8).sum(-1).max()<=3
        ((y-X)**2).mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

def test_shared_same_budget():
    sizes=[sum(p.numel() for p in SAE(m).parameters()) for m in METHODS[:-1]]
    assert sizes==[2624]*4
    assert sum(p.numel() for p in SAE('dense_onepass').parameters())==4192

def test_maps_rotation_is_identity_if_linear():
    _,_,_,H=fixed_maps()
    assert torch.allclose(H@H.T,torch.eye(32),atol=2e-6)
    m=SAE('mirror_loop')
    with torch.no_grad():
        aff=m.encoder_matrix(nonlinear_override='affine_loop')
        restored=m.encoder_matrix(nonlinear_override='linear_rot_invert')
    assert torch.allclose(aff,restored,atol=1e-6)

def test_affine_rank_vs_direct_nonlinear():
    torch.manual_seed(91)
    m=SAE('mirror_loop')
    with torch.no_grad():
        aff=spectrum(m.encoder_matrix(nonlinear_override='affine_loop'))
        non=spectrum(m.encoder_matrix(nonlinear_override='tanh_plain_loop'))
        mirrored=spectrum(m.encoder_matrix(nonlinear_override='mirror_loop'))
    assert aff['rank_rel_1e3']<=8 and non['rank_rel_1e3']>8 and mirrored['rank_rel_1e3']>8

def test_split_and_teacher():
    for world in WORLDS:
        w=teacher(101,world)
        assert w.shape==(64,32) and torch.allclose(w.norm(dim=1),torch.ones(64),atol=2e-6)
        a,_=dataset(101,world,'train',90)
        b,_=dataset(101,world,'dev',90)
        assert not torch.equal(a,b)
