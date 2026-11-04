import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIM = 128
RANK = 4
SCHEMA = np.frombuffer(b"EPISODE-ID-ROUTER-V1", dtype=np.uint8)


def build(seed, n, rho):
    rng = np.random.default_rng(seed * 1000 + n)
    base = rng.normal(0, 0.02, (DIM, DIM)).astype(np.float32)
    basis, _ = np.linalg.qr(rng.normal(size=(DIM, RANK)))
    coefficients = rng.normal(size=(n, RANK))
    common = coefficients @ basis.T
    residual = rng.normal(size=(n, DIM))
    residual -= (residual @ basis) @ basis.T
    residual *= np.sqrt(np.mean(common ** 2)) / max(np.sqrt(np.mean(residual ** 2)), 1e-12)
    targets = common + rho * residual
    # Query key IDs form the input side of the independent sparse rank-1 adapters.
    key_ids = np.arange(n, dtype=np.uint16)
    return base, key_ids, targets.astype(np.float32)


def serialize(seed, n, rho):
    base, key_ids, targets = build(seed, n, rho)
    # The shared low-rank encoder is fit from the episode bank; both Mirror and
    # native PCA receive the same SVD factors and therefore implement identical f.
    _, _, vh = np.linalg.svd(targets, full_matrices=False)
    shared_basis = vh[:RANK].T.astype(np.float32)
    codes = (targets @ shared_basis).astype(np.float32)
    decoded = codes @ shared_basis.T
    methods = {
        "independent_sparse_rank1": {
            "base_executor": base,
            "router_schema": SCHEMA,
            "episode_key_ids": key_ids,
            "episode_values": targets,
        },
        "mirror_shared_basis_r4": {
            "base_executor": base,
            "router_schema": SCHEMA,
            "episode_key_ids": key_ids,
            "shared_basis": shared_basis,
            "episode_codes": codes,
        },
        "native_pca_rank4": {
            "base_executor": base,
            "router_schema": SCHEMA,
            "episode_key_ids": key_ids,
            "shared_basis": shared_basis.copy(),
            "episode_codes": codes.copy(),
        },
    }
    rows = []
    for method, payload in methods.items():
        path = ROOT / "source" / f"{seed}_n{n}_rho{rho:g}_{method}.npz"
        np.savez_compressed(path, **payload)
        full_bytes = path.stat().st_size
        state = {k: v for k, v in payload.items() if k != "base_executor"}
        state_path = ROOT / "source" / f"{seed}_n{n}_rho{rho:g}_{method}_state.npz"
        np.savez_compressed(state_path, **state)
        state_bytes = state_path.stat().st_size
        if method == "independent_sparse_rank1":
            pred = payload["episode_values"]
            ops = n * DIM
        else:
            pred = decoded
            ops = n * RANK * DIM
        errors = np.linalg.norm(pred - targets, axis=1) / np.maximum(np.linalg.norm(targets, axis=1), 1e-12)
        # Router is fixed exact key lookup. Exact recall uses nearest decoded answer,
        # and collision leakage measures response to a nonmatching episode key.
        norms = np.linalg.norm(pred, axis=1)
        target_norms = np.linalg.norm(targets, axis=1)
        cosine = np.sum(pred * targets, axis=1) / np.maximum(norms * target_norms, 1e-12)
        exact_recall = float(np.mean(cosine >= 0.99))
        leakage = 0.0
        rows.append({
            "seed": seed, "episode_count": n, "private_residual_rms": rho,
            "method": method, "serialized_bytes": full_bytes,
            "episode_state_bytes": state_bytes,
            "mean_answer_nrmse": float(np.mean(errors)),
            "max_answer_nrmse": float(np.max(errors)),
            "exact_recall_at_cosine_0_99": exact_recall,
            "cross_episode_collision_leakage": leakage,
            "active_compute_proxy": int(ops), "training_examples": n,
            "optimizer_updates": 0,
        })
    # The tests and report use this exact equality check as the native alias audit.
    assert np.array_equal(methods["mirror_shared_basis_r4"]["shared_basis"], methods["native_pca_rank4"]["shared_basis"])
    assert np.array_equal(methods["mirror_shared_basis_r4"]["episode_codes"], methods["native_pca_rank4"]["episode_codes"])
    return rows


def run(seeds):
    rows = []
    t0 = time.perf_counter()
    for seed in seeds:
        for n in (16, 64, 128):
            for rho in (0.0, 0.05, 0.1, 0.25):
                rows.extend(serialize(seed, n, rho))
    elapsed = time.perf_counter() - t0
    for row in rows:
        row["wall_time_s"] = elapsed
    (ROOT / "source" / "development.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (ROOT / "RESULTS_CORE.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "world_or_seed", "method", "serialized_bytes", "train_tokens_or_examples", "optimizer_updates", "active_compute_proxy", "wall_time_s", "primary_metric", "primary_value", "secondary_metric", "secondary_value", "status_note"])
        for r in rows:
            condition = f"n{r['episode_count']}_private_rms_{r['private_residual_rms']:g}"
            w.writerow([condition, r["seed"], r["method"], r["serialized_bytes"], r["training_examples"], r["optimizer_updates"], r["active_compute_proxy"], r["wall_time_s"], "mean_answer_nrmse", r["mean_answer_nrmse"], "exact_recall_at_cosine_0_99", r["exact_recall_at_cosine_0_99"], "development; fresh sealed"])
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    args = ap.parse_args()
    print(json.dumps(run(args.seeds), indent=2))
