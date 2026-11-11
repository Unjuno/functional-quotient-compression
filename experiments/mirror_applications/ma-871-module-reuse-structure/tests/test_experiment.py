import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from experiment import evaluate, make_world, run, serialize_payload, fit_structure


def test_native_pca_and_mirror_are_exact_function_aliases():
    result = evaluate(809)
    for n in (4, 8, 16, 32):
        row = result["by_support"][str(n)]
        assert row["mse"]["mirror"] == row["mse"]["native_pca"]
        assert row["payload_bytes"]["mirror"] == row["payload_bytes"]["native_pca"]
        assert row["per_task_mse"]["mirror"] == row["per_task_mse"]["native_pca"]


def test_rank2_payload_smaller_than_independent_and_charges_all_codes():
    result = evaluate(809)
    sizes = result["payload_bytes_at_8"]
    assert sizes["mirror"] == 408
    assert sizes["independent"] == 488
    assert sizes["mirror"] < sizes["independent"]


def test_development_screen_keeps_random_basis_worse_and_tracks_all_shots():
    result = run([809])
    assert set(result["mean_mse_by_support"]) == {"4", "8", "16", "32"}
    assert result["mean_mse_by_support"]["8"]["mirror"] < result["mean_mse_by_support"]["8"]["random_basis"]
    assert result["max_mse_by_support"]["8"]["mirror"] < 0.0025


def test_serialized_payload_roundtrip_is_deterministic():
    world = make_world(809)
    model = fit_structure(world)
    coeff = np.zeros((6, 2), dtype=np.float32)
    payload_a = serialize_payload(world, model, "mirror", coeff)
    payload_b = serialize_payload(world, model, "mirror", coeff)
    assert payload_a == payload_b
