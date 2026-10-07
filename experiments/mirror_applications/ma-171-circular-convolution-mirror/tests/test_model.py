import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from model import METHODS, ExpertViews, address_matrices
from run import world, targets


def test_all_method_payloads_restore_identical_outputs():
    x=torch.randn(32,16);role=torch.arange(32)%4
    for method in METHODS:
        torch.manual_seed(17101)
        model=ExpertViews(method,17191)
        payload=model.serialize()
        restored=ExpertViews.from_serialized(payload)
        model.eval();restored.eval()
        assert torch.equal(model(x,role),restored(x,role))
        assert len(payload)==model.serialized_payload_bytes()


def test_hrr_addresses_are_orthogonal_circulant_views():
    mats=address_matrices(17191,"hrr")
    eye=torch.eye(16).expand_as(mats)
    assert torch.max(torch.abs(mats.transpose(1,2)@mats-eye))<1e-5


def test_aligned_teacher_is_represented_by_shared_hrr_model():
    w=world(17101,"aligned_hrr")
    x=torch.randn(24,16);role=torch.arange(24)%4
    expected=targets(x,role,w,"aligned_hrr")
    torch.manual_seed(1);m=ExpertViews("hrr",w['address_seed'])
    with torch.no_grad():m.weight.copy_(w['w']);m.bias.copy_(w['b'])
    assert torch.max(torch.abs(m(x,role)-expected))<1e-6
