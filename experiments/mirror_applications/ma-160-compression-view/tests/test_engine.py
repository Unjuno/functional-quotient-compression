import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from engine import METHODS, build, deserialize, make_world, run_one, serialize


def test_all_serialized_methods_roundtrip_decoded_weights_exactly():
    targets, _, _, _ = make_world(16001, True)
    for method in METHODS:
        decoded, payload = build(method, targets)
        assert len(payload) > 0
        decoded_again, payload_again = build(method, targets)
        assert payload_again == payload
        assert torch.equal(decoded, decoded_again)


def test_payload_parser_consumes_every_byte_and_preserves_metadata():
    targets, _, _, _ = make_world(16002, True)
    _, payload = build("mirror_shared_residual", targets)
    method, entries = deserialize(payload)
    assert method == "mirror_shared_residual"
    assert set(entries) == {"base", "angles", "residual", "coeffs"}
    assert len(payload) == len(bytes(payload))


def test_result_reports_actual_bytes_and_zero_training():
    result = run_one("mirror_shared_residual", 16001, "aligned")
    assert result["payload_bytes"] > 0
    assert result["optimizer_updates"] == 0
    assert result["examples"] == 512
