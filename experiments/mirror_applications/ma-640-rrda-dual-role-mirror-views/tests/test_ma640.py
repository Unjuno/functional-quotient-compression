import importlib.util
import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "source"
SPEC = importlib.util.spec_from_file_location("ma640", SRC / "run_experiment.py")
ma640 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ma640)


def test_development_has_expected_rows_and_exact_rank2():
    rows = ma640.run(64001)
    assert len(rows) == 30
    for angle in (0, 15, 30, 60, 90):
        rank2 = [r for r in rows if r["role_angle_degrees"] == angle and
                 r["method"] in ("mirror_shared_basis_r2", "native_shared_lowrank_rank2")]
        assert len(rank2) == 2
        assert all(r["edit_role_nrmse"] < 1e-6 and r["suppress_role_nrmse"] < 1e-6 for r in rank2)
        assert rank2[0]["serialized_bytes"] == rank2[1]["serialized_bytes"]


def test_rank1_mirror_aliases_native_gate_at_aligned_roles():
    rows = ma640.run(64001)
    aligned = [r for r in rows if r["role_angle_degrees"] == 0 and
               r["method"] in ("mirror_shared_basis_r1", "native_gate_rank1")]
    assert len(aligned) == 2
    assert abs(aligned[0]["edit_role_nrmse"] - aligned[1]["edit_role_nrmse"]) < 1e-12
    assert abs(aligned[0]["suppress_role_nrmse"] - aligned[1]["suppress_role_nrmse"]) < 1e-12


def test_serialized_payload_roundtrip_and_metric_replay():
    data = json.loads((SRC / "development.json").read_text())
    for row in data:
        path = SRC / f"{row['seed']}_theta{row['role_angle_degrees']}_{row['method']}.npz"
        with np.load(path) as payload:
            assert payload["base_executor"].shape == (16, 16)
            assert int(payload["role_ids"].size) == 2
            if row["method"] == "independent_role_adapters":
                pred = payload["role_values"]
                target = pred
            else:
                pred = payload["role_codes"] @ payload["shared_basis"].T
                # Targets are deterministically regenerated from the seed.
                rng = np.random.default_rng(row["seed"])
                _base = rng.normal(0, .1, (16, 16)).astype(np.float32)
                u = rng.normal(size=16); u /= np.linalg.norm(u)
                v = rng.normal(size=16); v -= u * (u @ v); v /= np.linalg.norm(v)
                th = np.deg2rad(row["role_angle_degrees"])
                target = np.stack([u, -np.cos(th) * u + np.sin(th) * v])
            errs = np.linalg.norm(pred - target, axis=1) / np.linalg.norm(target, axis=1)
            assert abs(float(errs[0]) - row["edit_role_nrmse"]) < 1e-6
            assert abs(float(errs[1]) - row["suppress_role_nrmse"]) < 1e-6
        assert path.stat().st_size == row["serialized_bytes"]
