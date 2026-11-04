import json
from pathlib import Path

import numpy as np

from run_experiment import ROOT, make_world


def load_rows():
    return json.loads((ROOT / "source" / "development.json").read_text())


def test_development_grid_and_actual_checkpoint_sizes():
    rows = load_rows()
    assert len(rows) == 32
    for row in rows:
        seed, k, method = row["seed"], row["distinct_gradient_modes"], row["method"]
        resume = ROOT / "source" / f"{seed}_k{k}_{method}_resume.npz"
        infer = ROOT / "source" / f"{seed}_k{k}_{method}_inference.npz"
        geometry = ROOT / "source" / f"{seed}_k{k}_{method}_geometry.npz"
        assert resume.stat().st_size == row["resume_payload_bytes"]
        assert infer.stat().st_size == row["inference_payload_bytes"]
        assert geometry.stat().st_size == row["geometry_payload_bytes"]
        assert row["optimizer_updates"] == 16 * 100


def test_mirror_native_alias_and_checkpoint_roundtrip():
    rows = load_rows()
    for seed in (64301, 64302):
        for k in (1, 2, 4, 8):
            mpath = ROOT / "source" / f"{seed}_k{k}_mirror_codebook_resume.npz"
            npath = ROOT / "source" / f"{seed}_k{k}_native_codebook_id_resume.npz"
            with np.load(mpath) as m, np.load(npath) as n:
                for key in ("weights", "first_moments", "second_moments", "projector_dictionary", "task_codes", "step_counts", "task_ids"):
                    assert np.array_equal(m[key], n[key])
            rm = next(x for x in rows if x["seed"] == seed and x["distinct_gradient_modes"] == k and x["method"] == "mirror_codebook")
            rn = next(x for x in rows if x["seed"] == seed and x["distinct_gradient_modes"] == k and x["method"] == "native_codebook_id")
            assert rm["mean_heldout_mse"] == rn["mean_heldout_mse"]
            assert rm["resume_payload_bytes"] == rn["resume_payload_bytes"]


def test_heldout_metric_replay():
    rows = load_rows()
    for row in rows:
        seed, k, method = row["seed"], row["distinct_gradient_modes"], row["method"]
        task_data = make_world(seed, k)
        path = ROOT / "source" / f"{seed}_k{k}_{method}_inference.npz"
        with np.load(path) as payload:
            weights = payload["weights"]
        losses = []
        for i, (_mode, _x, _y, xh, yh) in enumerate(task_data):
            losses.append(float(np.mean((xh @ weights[i] - yh) ** 2)))
        assert abs(float(np.mean(losses)) - row["mean_heldout_mse"]) < 1e-10
        assert abs(float(np.max(losses)) - row["max_heldout_mse"]) < 1e-10
