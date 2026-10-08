import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from model import METHODS, ExpertViews, phase_matrices
from run import world, targets


def test_all_method_payloads_restore_identical_outputs():
    x=torch.randn(32,16);role=torch.arange(32)%4
    for method in METHODS:
        torch.manual_seed(17301)
        model=ExpertViews(method,17391)
        payload=model.serialize();restored=ExpertViews.from_serialized(payload)
        model.eval();restored.eval()
        assert torch.equal(model(x,role),restored(x,role))
        assert len(payload)==model.serialized_payload_bytes()


def test_fourier_phase_views_are_real_orthogonal_operators():
    mats=phase_matrices(17391);eye=torch.eye(16).expand_as(mats)
    assert torch.max(torch.abs(mats.transpose(1,2)@mats-eye))<1e-5


def test_learned_phase_model_can_reconstruct_aligned_teacher():
    seed=17301;w=world(seed,"aligned_fft_phase")
    g=torch.Generator().manual_seed(seed+1900)
    phases=[]
    for _ in range(4):
        p=torch.rand(9,generator=g)*2*torch.pi
        phases.append(p[1:-1])
    x=torch.randn(24,16);role=torch.arange(24)%4;expected=targets(x,role,w,"aligned_fft_phase")
    torch.manual_seed(1);m=ExpertViews("fft_phase",w['address_seed'])
    with torch.no_grad():
        m.weight.copy_(w['w']);m.bias.copy_(w['b']);m.phase_inner.copy_(torch.stack(phases))
    assert torch.max(torch.abs(m(x,role)-expected))<3e-6
