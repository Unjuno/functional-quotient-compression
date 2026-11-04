import json
from pathlib import Path

import numpy as np

from run_experiment import ROOT, make_world, butterfly_view, SCALES


def test_shared_rotation_recovers_rank_but_static_basis_does_not():
    rows = json.loads((ROOT / "source" / "development.json").read_text())
    assert len(rows) == 10
    for seed in (64701, 64702):
        by = {r["method"]: r for r in rows if r["seed"] == seed}
        for name in ("relora_independent", "relora_independent_boft", "mirror_shared_butterfly_scalar", "native_shared_angle_boft"):
            assert by[name]["relative_frobenius_error"] < 1e-5
            assert by[name]["effective_update_rank"] == 16
        assert by["static_rank4_basis"]["effective_update_rank"] == 4
        assert by["static_rank4_basis"]["relative_frobenius_error"] > .7
        assert by["mirror_shared_butterfly_scalar"]["serialized_bytes"] == by["native_shared_angle_boft"]["serialized_bytes"]


def test_serialized_bytes_and_function_replay():
    rows = json.loads((ROOT / "source" / "development.json").read_text())
    for row in rows:
        path = ROOT / "source" / f"{row['seed']}_{row['method']}.npz"
        assert path.stat().st_size == row["serialized_bytes"]
        base, basis, theta, coeffs, views, phases, target = make_world(row["seed"])
        with np.load(path) as payload:
            assert np.array_equal(payload["base_weights"], base)
            if row["method"] == "relora_independent":
                decoded = (payload["phase_left_factors"] @ payload["phase_right_factors"]).sum(axis=0)
            elif row["method"] == "relora_independent_boft":
                decoded = sum((butterfly_view(payload["phase_angle_tables"][i], 1.0, payload["seed_basis"]) @ payload["phase_coefficients"][i] for i in range(4)), np.zeros((32, 32), dtype=np.float32))
            elif row["method"] in ("mirror_shared_butterfly_scalar", "native_shared_angle_boft"):
                decoded = sum((butterfly_view(payload["shared_butterfly_angles"], float(s), payload["seed_basis"]) @ payload["phase_coefficients"][i] for i, s in enumerate(payload["phase_view_scales"])), np.zeros((32, 32), dtype=np.float32))
            else:
                decoded = payload["seed_basis"] @ payload["phase_coefficients"].sum(axis=0)
        err = float(np.linalg.norm(decoded - target) / np.linalg.norm(target))
        assert abs(err - row["relative_frobenius_error"]) < 1e-6


def test_mirror_and_native_boft_are_exact_aliases():
    for seed in (64701, 64702):
        a = ROOT / "source" / f"{seed}_mirror_shared_butterfly_scalar.npz"
        b = ROOT / "source" / f"{seed}_native_shared_angle_boft.npz"
        with np.load(a) as x, np.load(b) as y:
            for key in ("base_weights", "seed_basis", "shared_butterfly_angles", "phase_view_scales", "phase_coefficients"):
                assert np.array_equal(x[key], y[key])
