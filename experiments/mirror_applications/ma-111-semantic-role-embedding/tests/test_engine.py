import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from engine import METHODS, Student, deserialize_records, make_batch, serialize_records, teacher_logits, world


def test_exact_teacher_state_matches_teacher_for_both_families():
    for condition in ('aligned', 'independent'):
        w = world(11101, condition)
        s = Student('exact_teacher', w, 3, .003)
        s.set_teacher_reference(w)
        x, _ = make_batch(w, 7, 32, True)
        for role in range(4):
            assert torch.max(torch.abs(s.forward(x, role)-teacher_logits(x, w, role))).item() < 1e-6


def test_all_method_payloads_round_trip_after_role_adaptation():
    w = world(11102, 'aligned')
    sdata, _ = make_batch(w, 8, 16, True)
    target = teacher_logits(sdata, w, 1)
    for method in METHODS:
        s = Student(method, w, 4, .003)
        s.set_teacher_reference(w)
        s.acquire(1, sdata, target, .003, updates=1)
        payload = s.serialize()
        name, records = deserialize_records(payload)
        assert serialize_records(name, records) == payload
