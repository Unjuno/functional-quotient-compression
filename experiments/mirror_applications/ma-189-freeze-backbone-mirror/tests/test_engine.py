import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from engine import D, METHODS, Learner, teacher, world


def test_all_inference_payloads_roundtrip_identical_predictions():
    w = world(18901, 'aligned'); x = torch.randn(32, D)
    for method in METHODS:
        m = Learner(method, w['base_w'], w['base_b'], 18902, .003)
        if method != 'hard_tie':
            m.fit(x[:8], teacher(x[:8], w), updates=2)
        payload = m.serialize(); restored = Learner.from_payload(payload)
        assert restored.serialize() == payload
        assert torch.equal(m.forward(x), restored.forward(x))


def test_two_sided_teacher_is_representable_by_two_angle_view():
    w = world(18903, 'aligned'); m = Learner('mirror_two_sided', w['base_w'], w['base_b'], 18904, .01)
    with torch.no_grad():
        m.angle_in.copy_(w['in_angle']); m.angle_out.copy_(w['out_angle'])
    x = torch.randn(64, D)
    assert torch.allclose(m.forward(x), teacher(x, w), atol=1e-6, rtol=1e-6)


def test_hypernetwork_has_nonzero_factor_gradients_at_initialization():
    w = world(18905, 'aligned'); m = Learner('hypernet_rank2', w['base_w'], w['base_b'], 18906, .003)
    x = torch.randn(16, D); y = teacher(x, w)
    (m.forward(x)-y).square().mean().backward()
    grads = [p.grad for p in m.hyper.parameters()]
    assert any(g is not None and torch.count_nonzero(g) for g in grads)


def test_incremental_and_resume_payload_bytes_are_nonnegative():
    w = world(18907, 'aligned'); m = Learner('mirror_two_sided', w['base_w'], w['base_b'], 18908, .003)
    m.fit(torch.randn(16, D), torch.randn(16, 8), updates=2)
    assert m.inference_bytes() >= m.base_bytes()
    assert m.resume_bytes() >= m.resume_base_bytes()
