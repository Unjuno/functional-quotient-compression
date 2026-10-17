"""MNS-001 preregistration-independent L0 math and causal-leakage checks."""
import io
import math
import torch
from torch.nn import functional as F
from source.mechanism import (SharedMirrorCausal, rotate_first_plane,
                       conjugated_relu, inference_payload_bytes)


def test_common_qk_rotation_is_a_score_gauge():
    torch.manual_seed(123)
    q, k = torch.randn(3, 11, 8), torch.randn(3, 11, 8)
    a = torch.tensor(0.73)
    plain = q @ k.transpose(-1, -2)
    mirror = rotate_first_plane(q, a) @ rotate_first_plane(k, a).transpose(-1, -2)
    assert (plain - mirror).abs().max().item() < 3e-6


def test_different_qk_rotations_change_attention_scores():
    torch.manual_seed(124)
    q, k = torch.randn(3, 11, 8), torch.randn(3, 11, 8)
    plain = q @ k.transpose(-1, -2)
    mirror = rotate_first_plane(q, torch.tensor(0.73)) @ rotate_first_plane(k, torch.tensor(-0.29)).transpose(-1, -2)
    assert (plain - mirror).abs().mean().item() > 0.05


def test_view_output_has_no_future_token_leakage():
    torch.manual_seed(125)
    model = SharedMirrorCausal(8, 4).eval()
    a = torch.randn(2, 9, 8)
    b = a.clone()
    b[:, 6:, :] = torch.randn_like(b[:, 6:, :]) * 50
    with torch.no_grad():
        y1, y2 = model(a), model(b)
    assert (y1[:, :, :6, :] - y2[:, :, :6, :]).abs().max().item() < 1e-6
    assert (y1[:, :, 6:, :] - y2[:, :, 6:, :]).abs().max().item() > 0.01


def test_parallel_mirror_views_are_functionally_distinct():
    torch.manual_seed(126)
    model = SharedMirrorCausal(8, 4)
    y = model(torch.randn(3, 7, 8))
    assert y.shape == (3, 4, 7, 8)
    assert (y[:, 0] - y[:, 1]).abs().mean().item() > 1e-3


def test_later_head_angle_receives_gradient():
    torch.manual_seed(127)
    model = SharedMirrorCausal(8, 4)
    y = model(torch.randn(2, 5, 8))
    loss = (y * torch.randn_like(y)).sum()
    loss.backward()
    assert torch.all(torch.isfinite(model.q_angle.grad))
    assert torch.all(torch.isfinite(model.k_angle.grad))
    assert model.q_angle.grad.abs().sum().item() > 1e-4
    assert model.k_angle.grad.abs().sum().item() > 1e-4
    assert model.v_angle.grad.abs().sum().item() > 1e-4


def test_conjugated_relu_nontrivial():
    torch.manual_seed(128)
    z = torch.randn(20, 8)
    q = conjugated_relu(z, torch.tensor(0.49))
    assert (q - F.relu(z)).abs().mean().item() > 1e-3


def test_same_relu_sandwich_is_idempotent_and_not_extra_depth():
    torch.manual_seed(129)
    z = torch.randn(10, 8)
    angle = torch.tensor(0.49)
    once = conjugated_relu(z, angle)
    twice = conjugated_relu(once, angle)
    assert (once - twice).abs().max().item() < 5e-7


def test_residual_recurrent_mirror_has_real_multiple_steps():
    torch.manual_seed(130)
    z, w = torch.randn(10, 8), torch.randn(8, 8) / 4
    angle = torch.tensor(0.49)
    def step(h):
        return h + 0.25 * conjugated_relu(h @ w, angle)
    first = step(z)
    second = step(first)
    assert (first - second).abs().mean().item() > 1e-3


def test_roundtrip_and_accounted_bytes():
    torch.manual_seed(131)
    model = SharedMirrorCausal(8, 4)
    x = torch.randn(2, 9, 8)
    buf=io.BytesIO()
    torch.save({'cfg': {'d':8,'views':4}, 'state':model.state_dict()},buf)
    assert inference_payload_bytes(model)==len(buf.getvalue())
    obj=torch.load(io.BytesIO(buf.getvalue()),weights_only=True)
    clone=SharedMirrorCausal(**obj['cfg']);clone.load_state_dict(obj['state'])
    with torch.no_grad():
        assert torch.equal(model(x),clone(x))
    assert sum(p.numel() for p in model.parameters()) == 3*8*8+3*4


def test_parallel_views_do_not_make_recurrence_parallel():
    torch.manual_seed(132)
    model=SharedMirrorCausal(8,4)
    x=torch.randn(1,5,8)
    y0=model(x).mean(dim=1)
    y1=model(x+0.25*y0).mean(dim=1)
    assert (y0-y1).abs().mean().item() > 1e-3
