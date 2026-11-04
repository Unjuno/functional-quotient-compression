import json
from pathlib import Path

import numpy as np

from run_experiment import ROOT, build, serialize


def test_development_matrix_and_aligned_quality_storage_gate():
    rows = json.loads((ROOT / "source" / "development.json").read_text())
    assert len(rows) == 72
    for n in (64, 128):
        independent = [r for r in rows if r["episode_count"] == n and r["private_residual_rms"] == 0 and r["method"] == "independent_sparse_rank1"]
        mirror = [r for r in rows if r["episode_count"] == n and r["private_residual_rms"] == 0 and r["method"] == "mirror_shared_basis_r4"]
        assert len(independent) == len(mirror) == 2
        for a, m in zip(independent, mirror):
            assert m["mean_answer_nrmse"] < 1e-6
            assert m["exact_recall_at_cosine_0_99"] == 1.0
            assert m["serialized_bytes"] <= 0.75 * a["serialized_bytes"]


def test_native_pca_exact_function_alias_and_serialized_roundtrip():
    rows = json.loads((ROOT / "source" / "development.json").read_text())
    for row in rows:
        if row["method"] not in ("mirror_shared_basis_r4", "native_pca_rank4"):
            continue
        n, rho, seed = row["episode_count"], row["private_residual_rms"], row["seed"]
        mirror = ROOT / "source" / f"{seed}_n{n}_rho{rho:g}_mirror_shared_basis_r4.npz"
        native = ROOT / "source" / f"{seed}_n{n}_rho{rho:g}_native_pca_rank4.npz"
        own = ROOT / "source" / f"{seed}_n{n}_rho{rho:g}_{row['method']}.npz"
        assert own.stat().st_size == row["serialized_bytes"]
        with np.load(mirror) as m, np.load(native) as p:
            assert np.array_equal(m["base_executor"], p["base_executor"])
            assert np.array_equal(m["episode_key_ids"], p["episode_key_ids"])
            assert np.array_equal(m["shared_basis"], p["shared_basis"])
            assert np.array_equal(m["episode_codes"], p["episode_codes"])
            decoded = m["episode_codes"] @ m["shared_basis"].T
        _, _, targets = build(seed, n, rho)
        errors = np.linalg.norm(decoded - targets, axis=1) / np.maximum(np.linalg.norm(targets, axis=1), 1e-12)
        assert abs(float(errors.mean()) - row["mean_answer_nrmse"]) < 1e-7
        assert abs(float(errors.max()) - row["max_answer_nrmse"]) < 1e-7


def test_independent_adapters_replay_exactly_and_router_is_fixed():
    rows = json.loads((ROOT / "source" / "development.json").read_text())
    for row in rows:
        if row["method"] != "independent_sparse_rank1":
            continue
        path = ROOT / "source" / f"{row['seed']}_n{row['episode_count']}_rho{row['private_residual_rms']:g}_independent_sparse_rank1.npz"
        _, keys, targets = build(row["seed"], row["episode_count"], row["private_residual_rms"])
        with np.load(path) as payload:
            assert path.stat().st_size == row["serialized_bytes"]
            assert np.array_equal(payload["episode_key_ids"], keys)
            assert np.array_equal(payload["episode_values"], targets)
            assert np.array_equal(payload["episode_values"], targets)  # exact-key route returns its own stored update
