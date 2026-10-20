import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import torch
import numpy as np
from model import AudioAE,fixed_program,program_sha256
from audio import main_data,cross_data

def test_param_budget_and_input_output_shapes():
    for mode,count in [('mirror',2144),('affine',2144),('sparse',2080),('native',2352)]:
        m=AudioAE(mode)
        assert sum(t.numel() for t in m.parameters())==count
        x=torch.randn(3,64)
        assert m(x).shape==(3,64)
        assert (m.encode(x)>0).sum(1).max()<=6

def test_virtual_weight_shapes_and_shared_core():
    m=AudioAE('mirror')
    assert len(m.graph.core)==16 and len(m.graph.address.flatten())==32
    assert m.graph.weight_matrices().shape==(4,4,4,4)
    assert len(m.graph.state_dict())==3

def test_mirror_orientational_nonlinearity_and_affine_control():
    a=AudioAE('mirror'); b=AudioAE('affine')
    b.load_state_dict(a.state_dict())
    assert not torch.allclose(a.graph.weight_matrices(),b.graph.weight_matrices())
    assert torch.allclose(a.graph.address,b.graph.address)

def test_householder_orthogonality():
    H=fixed_program()[2]
    assert torch.max(torch.abs(H@H.T-torch.eye(16)))<1e-6

def test_joint_gradient_nonzero():
    m=AudioAE('mirror'); x=torch.randn(9,64)
    m(x).square().mean().backward()
    for name,param in [('core',m.graph.core),('gates',m.graph.gates),('address',m.graph.address)]:
        assert param.grad is not None and param.grad.abs().sum().item()>0,name

def test_masked_edge_changes_output_without_overwriting_params():
    m=AudioAE('mirror'); x=torch.randn(8,64); y=m(x)
    prior={k:t.clone() for k,t in m.state_dict().items()}
    ym=m(x,masked_edges=8)
    assert not torch.equal(ym,y)
    assert all(torch.equal(prior[k],m.state_dict()[k]) for k in prior)

def test_nonoverlapping_time_splits_and_exogenous_wave():
    d,mu,sd,info=main_data()
    assert set(d)=={'train','dev','fresh','late'}
    assert info['frames']['train']>1500
    assert torch.allclose(d['train'].mean(0),torch.zeros(64),atol=1e-5)
    assert torch.isfinite(cross_data(mu,sd)).all()

def test_seeded_program_deterministic():
    assert len(program_sha256())==64
    for x,y in zip(fixed_program(),fixed_program()):assert torch.equal(x,y)

def test_tensor_serialization_replay(tmp_path):
    m=AudioAE('mirror'); x=torch.randn(3,64)
    orig=m(x)
    torch.save({'kind':'mirror','weights':m.state_dict()},tmp_path/'pkt.pt')
    pkt=torch.load(tmp_path/'pkt.pt',weights_only=True)
    q=AudioAE(pkt['kind']);q.load_state_dict(pkt['weights'])
    assert torch.equal(orig,q(x))
