import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from engine import METHODS, Student, deserialize_records, effective_teacher_state, serialize_records, teacher_logits, world


def test_record_payload_round_trip_is_byte_exact():
    w = world(20801, 'aligned')
    student = Student('mirror_view', w['base'], 1, .003)
    student.angles[1] = torch.nn.Parameter(torch.tensor(.125))
    payload = student.serialize()
    method, records = deserialize_records(payload)
    assert method == 'mirror_view'
    assert serialize_records(method, records) == payload


def test_aligned_teacher_can_be_written_as_ordinary_mlp():
    w = world(20801, 'aligned')
    state = effective_teacher_state(w, 2)
    x = torch.randn(128, 12, generator=torch.Generator().manual_seed(99))
    logits = teacher_logits(x, w, 2)
    w1, b1, w2, b2 = state
    reconstructed = F.relu(x @ w1 + b1) @ w2 + b2
    assert torch.max(torch.abs(logits - reconstructed)).item() < 2e-6


def test_all_student_payload_variants_round_trip_after_adaptation():
    w = world(20802, 'aligned')
    x = torch.randn(16, 12, generator=torch.Generator().manual_seed(88))
    logits = teacher_logits(x, w, 1)
    for method in METHODS:
        student = Student(method, w['base'], 7, .003)
        student.acquire(1, x, logits, .003, updates=1, teacher_state=effective_teacher_state(w, 1))
        payload = student.serialize()
        parsed_method, records = deserialize_records(payload)
        assert serialize_records(parsed_method, records) == payload
