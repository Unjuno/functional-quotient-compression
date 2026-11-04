import json
from pathlib import Path

import numpy as np

from run_experiment import ROOT, make_target


def test_rank_growth_boundary_and_native_dictionary_alias():
    rows = json.loads((ROOT / "source" / "development.json").read_text())
    assert len(rows) == 8
    for seed in (64601, 64602):
        by = {r["method"]: r for r in rows if r["seed"] == seed}
        assert by["relora_merged"]["effective_update_rank"] == 16
        assert by["relora_merged"]["relative_frobenius_error"] < 1e-5
        assert by["mirror_static_basis"]["effective_update_rank"] <= 4
        assert by["mirror_static_basis"]["relative_frobenius_error"] > .5
        assert by["mirror_dictionary_code"]["effective_update_rank"] == 16
        assert by["mirror_dictionary_code"]["relative_frobenius_error"] < 1e-5
        assert by["mirror_dictionary_code"]["restart_state_bytes"] == by["native_dictionary_block_id"]["restart_state_bytes"]
        assert by["mirror_dictionary_code"]["relative_frobenius_error"] == by["native_dictionary_block_id"]["relative_frobenius_error"]


def test_payload_roundtrip_and_actual_byte_counts():
    rows = json.loads((ROOT / "source" / "development.json").read_text())
    for row in rows:
        seed, method = row["seed"], row["method"]
        ip = ROOT / "source" / f"{seed}_{method}_inference.npz"
        rp = ROOT / "source" / f"{seed}_{method}_restart.npz"
        assert ip.stat().st_size == row["inference_payload_bytes"]
        assert rp.stat().st_size == row["restart_state_bytes"]
        base, _basis, _pu, _pv, target = make_target(seed)
        with np.load(ip) as payload:
            assert np.array_equal(payload["base_weights"], base)
            if method == "relora_merged":
                update = payload["merged_update"]
            elif method == "mirror_static_basis":
                update = payload["shared_basis"] @ payload["phase_coefficients"].sum(axis=0)
            else:
                update = sum((payload["shared_dictionary"][:, int(c) * 4:(int(c) + 1) * 4] @ payload["phase_coefficients"][i]
                              for i, c in enumerate(payload["phase_codes"])), np.zeros((32, 32), dtype=np.float32))
        rel = np.linalg.norm(update - target) / np.linalg.norm(target)
        assert abs(float(rel) - row["relative_frobenius_error"]) < 1e-6


def test_native_control_serializes_same_function_arrays():
    for seed in (64601, 64602):
        mp = ROOT / "source" / f"{seed}_mirror_dictionary_code_inference.npz"
        npth = ROOT / "source" / f"{seed}_native_dictionary_block_id_inference.npz"
        with np.load(mp) as m, np.load(npth) as n:
            for key in ("base_weights", "shared_dictionary", "phase_coefficients", "phase_codes"):
                assert np.array_equal(m[key], n[key])
