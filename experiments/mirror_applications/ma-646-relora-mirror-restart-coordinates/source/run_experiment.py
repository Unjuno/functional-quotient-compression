import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D, PHASES, RANK = 32, 4, 4
SCHEMA = np.frombuffer(b"RELORA-RESTART-V1", dtype=np.uint8)


def make_target(seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    u = q[:, :PHASES * RANK]
    v = rng.normal(0, .2, (PHASES * RANK, D))
    phase_u = np.stack([u[:, p * RANK:(p + 1) * RANK] for p in range(PHASES)])
    phase_v = np.stack([v[p * RANK:(p + 1) * RANK] for p in range(PHASES)])
    phase_updates = np.stack([phase_u[p] @ phase_v[p] for p in range(PHASES)])
    target = phase_updates.sum(axis=0)
    base = rng.normal(0, .02, (D, D)).astype(np.float32)
    return base, u.astype(np.float32), phase_u.astype(np.float32), phase_v.astype(np.float32), target.astype(np.float32)


def relerr(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def numerical_rank(a, tol=1e-5):
    s = np.linalg.svd(a, compute_uv=False)
    return int(np.sum(s > tol * s.max()))


def save(seed, method, inference, restart):
    ip = ROOT / "source" / f"{seed}_{method}_inference.npz"
    rp = ROOT / "source" / f"{seed}_{method}_restart.npz"
    t = time.perf_counter(); np.savez_compressed(ip, **inference); inference_s = time.perf_counter() - t
    t = time.perf_counter(); np.savez_compressed(rp, **restart); restart_s = time.perf_counter() - t
    return ip.stat().st_size, rp.stat().st_size, inference_s + restart_s


def run_seed(seed):
    base, full_basis, phase_u, phase_v, target = make_target(seed)
    # Ordinary ReLoRA control: independent low-rank factors per restart, merged at inference.
    relora_update = sum((phase_u[p] @ phase_v[p] for p in range(PHASES)), np.zeros((D, D), dtype=np.float32))
    relora_infer = {"schema": SCHEMA, "base_weights": base, "merged_update": relora_update}
    relora_restart = {"schema": SCHEMA, "base_weights": base, "phase_left_factors": phase_u, "phase_right_factors": phase_v,
                      "phase_ids": np.arange(PHASES, dtype=np.uint8)}

    # Static shared basis: all phase codes act in the same first rank-4 subspace.
    static_basis = full_basis[:, :RANK]
    static_coeffs = np.stack([static_basis.T @ (phase_u[p] @ phase_v[p]) for p in range(PHASES)])
    static_update = static_basis @ static_coeffs.sum(axis=0)
    static_infer = {"schema": SCHEMA, "base_weights": base, "shared_basis": static_basis, "phase_coefficients": static_coeffs}
    static_restart = {**static_infer, "restart_ids": np.arange(PHASES, dtype=np.uint8)}

    # Shared overcomplete dictionary: each restart code addresses one rank-4 block.
    coeffs = phase_v.copy()
    codes = np.arange(PHASES, dtype=np.uint8)
    decoded_phases = np.stack([full_basis[:, int(c) * RANK:(int(c) + 1) * RANK] @ coeffs[i] for i, c in enumerate(codes)])
    coded_update = decoded_phases.sum(axis=0)
    dict_infer = {"schema": SCHEMA, "base_weights": base, "shared_dictionary": full_basis,
                  "phase_coefficients": coeffs, "phase_codes": codes}
    native_infer = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in dict_infer.items()}
    dict_restart = dict_infer
    native_restart = native_infer

    methods = [
        ("relora_merged", relora_update, relora_infer, relora_restart),
        ("mirror_static_basis", static_update, static_infer, static_restart),
        ("mirror_dictionary_code", coded_update, dict_infer, dict_restart),
        ("native_dictionary_block_id", coded_update, native_infer, native_restart),
    ]
    rows = []
    for method, update, inference, restart in methods:
        ib, rb, wall = save(seed, method, inference, restart)
        rows.append({
            "seed": seed, "method": method, "inference_payload_bytes": ib, "restart_state_bytes": rb,
            "relative_frobenius_error": relerr(update, target), "effective_update_rank": numerical_rank(update),
            "target_rank": PHASES * RANK, "phase_count": PHASES, "optimizer_updates": 0,
            "reconstruction_mac_proxy": int(PHASES * 2 * D * RANK * D), "wall_time_s": wall,
        })
    assert np.array_equal(dict_infer["shared_dictionary"], native_infer["shared_dictionary"])
    assert np.array_equal(dict_infer["phase_codes"], native_infer["phase_codes"])
    assert np.array_equal(dict_infer["phase_coefficients"], native_infer["phase_coefficients"])
    return rows


def run(seeds):
    rows = [row for seed in seeds for row in run_seed(seed)]
    (ROOT / "source" / "development.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (ROOT / "RESULTS_CORE.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "world_or_seed", "method", "serialized_bytes", "train_tokens_or_examples", "optimizer_updates", "active_compute_proxy", "wall_time_s", "primary_metric", "primary_value", "secondary_metric", "secondary_value", "status_note"])
        for r in rows:
            w.writerow(["rank16_four_restart_updates", r["seed"], r["method"], r["inference_payload_bytes"], 0, 0, r["reconstruction_mac_proxy"], r["wall_time_s"], "relative_frobenius_error", r["relative_frobenius_error"], "effective_update_rank", r["effective_update_rank"], f"restart_state_bytes={r['restart_state_bytes']}; development; fresh sealed"])
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", nargs="+", type=int, required=True); args = ap.parse_args()
    print(json.dumps(run(args.seeds), indent=2))
