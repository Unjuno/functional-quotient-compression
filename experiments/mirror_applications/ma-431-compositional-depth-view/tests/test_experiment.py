import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import experiment as exp


def test_mirror_view_recovers_noncommuting_teacher_operators():
    shear = 0.3
    w = exp.op_matrix(0, shear)
    a = exp.mirror_matrix(w, 0)
    b = exp.mirror_matrix(w, 1)
    assert torch.allclose(a, exp.op_matrix(0, shear))
    assert torch.allclose(b, exp.op_matrix(1, shear))
    assert not torch.allclose(a @ b, b @ a)


def test_heldout_sequences_are_excluded_from_training():
    test = set(exp.heldout_programs())
    assert test
    assert all(len(p) == exp.MAX_LEN for p in test)
    assert not (test & set(exp.all_programs()))


def test_payload_charges_metadata_and_roundtrips():
    model = exp.make_model("mirror", 7)
    payload = exp.inference_payload(model)
    size, digest, replay = exp.payload_bytes(model)
    assert size == len(payload)
    assert digest == __import__("hashlib").sha256(payload).hexdigest()
    assert replay == payload
    decoded = torch.load(__import__("io").BytesIO(payload), weights_only=False)
    assert decoded["metadata"]["operation_codebook"] == {"0": 0.0, "1": __import__("math").pi / 2}
    assert len(payload) > sum(v.numel() * v.element_size() for v in decoded["state_dict"].values())


def test_control_payloads_and_compute_proxy_distinguish_costs():
    mirror = exp.make_model("mirror", 0)
    independent = exp.make_model("independent", 0)
    assert exp.payload_bytes(independent)[0] > exp.payload_bytes(mirror)[0]
    assert exp.active_macs_per_example("independent", 6) < exp.active_macs_per_example("mirror", 6)
