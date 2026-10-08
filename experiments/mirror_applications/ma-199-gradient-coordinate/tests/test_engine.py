import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from engine import D, METHODS, Learner, teacher, world


def test_all_method_inference_payloads_roundtrip_identically():
    w = world(19901, 'aligned'); x = torch.randn(24, D); y = teacher(x, w, 1)
    for method in METHODS:
        m = Learner(method, w['base_w'], w['base_b'], w['plane'], 19902, .003)
        m.acquire(1, x[:8], y[:8], .003, updates=2)
        payload = m.serialize(); restored = Learner.from_payload(payload)
        assert restored.serialize() == payload
        for task in (0, 1): assert torch.equal(m.forward(x, task), restored.forward(x, task))


def test_gradient_angle_plus_vector_reconstructs_aligned_rank_one_delta():
    w = world(19903, 'aligned'); m = Learner('mirror_gradient', w['base_w'], w['base_b'], w['plane'], 19904, .003)
    angle, vector = nn.Parameter(w['angles'][1].clone()), nn.Parameter(w['vectors'][1].clone())
    m.codes[1] = (angle, vector); m.seen.append(1)
    x = torch.randn(128, D)
    assert torch.allclose(m.forward(x, 1), teacher(x, w, 1), atol=1e-6, rtol=1e-6)


def test_resume_payload_includes_optimizer_states():
    w = world(19905, 'aligned'); m = Learner('mirror_gradient', w['base_w'], w['base_b'], w['plane'], 19906, .003)
    x = torch.randn(16, D); m.acquire(1, x, teacher(x, w, 1), .003, updates=3)
    assert m.resume_bytes() > len(m.serialize())
    assert m.resume_bytes() > m.resume_base_bytes()
