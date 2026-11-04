import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D, FAM, N_SUPPORT, N_QUERY, RANK = 32, 8, 256, 512, 4
SCHEMA = np.frombuffer(b"META-SGD-FAMILY-CODE-V1", dtype=np.uint8)


def make_world(seed, heterogeneity):
    rng = np.random.default_rng(seed * 100 + int(heterogeneity * 100))
    base_init = rng.normal(0, .01, size=D).astype(np.float32)
    patterns, _ = np.linalg.qr(rng.normal(size=(D, 3)))
    family_latents = rng.normal(size=(FAM, 3))
    common_log_curvature = .55 * family_latents @ patterns.T / np.sqrt(3)
    private = rng.normal(size=(FAM, D))
    loglam = common_log_curvature + heterogeneity * private
    lambdas = np.exp(loglam)
    train, query = [], []
    for f in range(FAM):
        ytr = rng.normal(size=(N_SUPPORT, D))
        gtr = lambdas[f] * ytr + .2 * rng.normal(size=(N_SUPPORT, D))
        yq = rng.normal(size=(N_QUERY, D))
        gq = lambdas[f] * yq + .2 * rng.normal(size=(N_QUERY, D))
        train.append((gtr, ytr)); query.append((gq, yq))
    return base_init, lambdas, train, query


def fit_alpha(g, y, lambdas):
    # Least-squares one-step Meta-SGD rule per coordinate.
    lam = np.broadcast_to(lambdas, g.shape)
    numerator = np.sum(lam * g * y, axis=0)
    denominator = np.sum(lam * g * g, axis=0) + 1e-12
    return numerator / denominator


def eval_loss(alpha, lambdas, query):
    losses = []
    for f, (g, y) in enumerate(query):
        pred_delta = alpha[f][None, :] * g
        losses.append(float(.5 * np.mean(lambdas[f][None, :] * (pred_delta - y) ** 2)))
    return np.asarray(losses)


def save(seed, hetero, name, payload):
    p = ROOT / "source" / f"{seed}_h{hetero:g}_{name}.npz"
    t = time.perf_counter(); np.savez_compressed(p, **payload); wall = time.perf_counter() - t
    return p.stat().st_size, wall


def run_condition(seed, hetero):
    base_init, lambdas, train, query = make_world(seed, hetero)
    family_alpha = np.stack([fit_alpha(*train[f], lambdas[f]) for f in range(FAM)])
    pooled_g = np.concatenate([x[0] for x in train], axis=0)
    pooled_y = np.concatenate([x[1] for x in train], axis=0)
    pooled_lambda = np.concatenate([np.repeat(lambdas[f][None, :], N_SUPPORT, axis=0) for f in range(FAM)], axis=0)
    global_alpha = fit_alpha(pooled_g, pooled_y, pooled_lambda)
    global_bank = np.repeat(global_alpha[None, :], FAM, axis=0)

    # Mirror and native PCA use exactly the same SVD factors and decoder.
    _, _, vh = np.linalg.svd(family_alpha, full_matrices=False)
    basis = vh[:RANK].T
    codes = family_alpha @ basis
    mirror_alpha = codes @ basis.T
    rng = np.random.default_rng(seed + 92000 + int(hetero * 100))
    random_basis, _ = np.linalg.qr(rng.normal(size=(D, RANK)))
    random_codes = family_alpha @ random_basis
    random_alpha = random_codes @ random_basis.T
    common = {"schema": SCHEMA, "base_initialization": base_init, "family_ids": np.arange(FAM, dtype=np.uint8)}
    methods = {
        "global_meta_sgd": (global_bank, {**common, "global_update_vector": global_alpha.astype(np.float32)}),
        "independent_family_meta_sgd": (family_alpha, {**common, "family_update_vectors": family_alpha.astype(np.float32)}),
        "mirror_pca_rank4": (mirror_alpha, {**common, "shared_basis": basis.astype(np.float32), "family_codes": codes.astype(np.float32)}),
        "native_pca_rank4": (mirror_alpha.copy(), {**common, "shared_basis": basis.astype(np.float32).copy(), "family_codes": codes.astype(np.float32).copy()}),
        "random_subspace_rank4": (random_alpha, {**common, "random_basis": random_basis.astype(np.float32), "family_codes": random_codes.astype(np.float32)}),
    }
    rows = []
    for name, (alpha_bank, payload) in methods.items():
        size, wall = save(seed, hetero, name, payload)
        losses = eval_loss(alpha_bank, lambdas, query)
        rows.append({
            "seed": seed, "heterogeneity": hetero, "method": name,
            "serialized_bytes": size, "mean_query_loss": float(np.mean(losses)),
            "max_family_query_loss": float(np.max(losses)), "family_count": FAM,
            "support_examples": FAM * N_SUPPORT, "query_examples": FAM * N_QUERY,
            "optimizer_updates": 0, "adaptation_mac_proxy": int(FAM * N_QUERY * D),
            "basis_decode_mac_proxy": int(FAM * D * RANK) if "rank4" in name else 0,
            "basis_fit_svd_count": int(name in ("mirror_pca_rank4", "native_pca_rank4")),
            "wall_time_s": wall,
        })
    assert np.array_equal(methods["mirror_pca_rank4"][1]["shared_basis"], methods["native_pca_rank4"][1]["shared_basis"])
    assert np.array_equal(methods["mirror_pca_rank4"][1]["family_codes"], methods["native_pca_rank4"][1]["family_codes"])
    assert np.array_equal(methods["mirror_pca_rank4"][0], methods["native_pca_rank4"][0])
    return rows


def run(seeds):
    rows = [r for seed in seeds for h in (0.0, .25, .5) for r in run_condition(seed, h)]
    (ROOT / "source" / "development.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (ROOT / "RESULTS_CORE.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "world_or_seed", "method", "serialized_bytes", "train_tokens_or_examples", "optimizer_updates", "active_compute_proxy", "wall_time_s", "primary_metric", "primary_value", "secondary_metric", "secondary_value", "status_note"])
        for r in rows:
            w.writerow([f"heterogeneity_{r['heterogeneity']:g}", r["seed"], r["method"], r["serialized_bytes"], r["support_examples"], r["optimizer_updates"], r["adaptation_mac_proxy"] + r["basis_decode_mac_proxy"], r["wall_time_s"], "mean_query_loss", r["mean_query_loss"], "max_family_query_loss", r["max_family_query_loss"], "development; fresh sealed"])
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", nargs="+", type=int, required=True); args = ap.parse_args()
    print(json.dumps(run(args.seeds), indent=2))
