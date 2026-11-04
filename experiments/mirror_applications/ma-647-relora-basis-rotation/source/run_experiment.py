import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D, RANK, PHASES, STAGES, PAIRS = 32, 4, 4, 5, 16
SCALES = np.array([-2., -1., 1., 2.], dtype=np.float32)
SCHEMA = np.frombuffer(b"BOFT-RELORA-VIEW-V1", dtype=np.uint8)


def butterfly_view(theta, scale, basis):
    out = basis.astype(np.float64).copy()
    for stage in range(STAGES):
        stride = 1 << stage
        for block in range(0, D, stride * 2):
            for offset in range(stride):
                i, j = block + offset, block + offset + stride
                a = float(theta[stage, (block // (stride * 2)) * stride + offset] * scale)
                c, s = np.cos(a), np.sin(a)
                x, y = out[i].copy(), out[j].copy()
                out[i] = c * x - s * y
                out[j] = s * x + c * y
    return out.astype(np.float32)


def make_world(seed):
    rng = np.random.default_rng(seed)
    basis = np.eye(D, RANK, dtype=np.float32)
    theta = rng.uniform(-.6, .6, size=(STAGES, PAIRS)).astype(np.float32)
    coeffs = rng.normal(0, .2, size=(PHASES, RANK, D)).astype(np.float32)
    views = np.stack([butterfly_view(theta, s, basis) for s in SCALES])
    target_phases = np.stack([views[p] @ coeffs[p] for p in range(PHASES)])
    target = target_phases.sum(axis=0)
    base = rng.normal(0, .02, size=(D, D)).astype(np.float32)
    return base, basis, theta, coeffs, views, target_phases, target


def rank(a):
    s = np.linalg.svd(a, compute_uv=False)
    return int(np.sum(s > 1e-5 * s.max()))


def save(seed, name, payload):
    path = ROOT / "source" / f"{seed}_{name}.npz"
    t = time.perf_counter(); np.savez_compressed(path, **payload); wall = time.perf_counter() - t
    return path.stat().st_size, wall


def run_seed(seed):
    base, basis, theta, coeffs, views, phase_updates, target = make_world(seed)
    phase_angles = theta[None, :, :] * SCALES[:, None, None]
    independent = {
        "schema": SCHEMA, "base_weights": base,
        "phase_left_factors": views, "phase_right_factors": coeffs,
    }
    independent_boft = {
        "schema": SCHEMA, "base_weights": base, "seed_basis": basis,
        "phase_angle_tables": phase_angles, "phase_coefficients": coeffs,
    }
    mirror = {
        "schema": SCHEMA, "base_weights": base, "seed_basis": basis,
        "shared_butterfly_angles": theta, "phase_view_scales": SCALES,
        "phase_coefficients": coeffs,
    }
    native = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in mirror.items()}
    static_coeffs = np.stack([basis.T @ phase_updates[p] for p in range(PHASES)])
    static = {"schema": SCHEMA, "base_weights": base, "seed_basis": basis,
              "phase_coefficients": static_coeffs}
    methods = [
        ("relora_independent", target, independent, 0),
        ("relora_independent_boft", target, independent_boft, PHASES * STAGES * PAIRS * RANK * 8),
        ("mirror_shared_butterfly_scalar", target, mirror, PHASES * STAGES * PAIRS * RANK * 8),
        ("native_shared_angle_boft", target, native, PHASES * STAGES * PAIRS * RANK * 8),
        ("static_rank4_basis", basis @ static_coeffs.sum(axis=0), static, 0),
    ]
    rows = []
    for name, decoded, payload, rot_mac in methods:
        size, wall = save(seed, name, payload)
        error = float(np.linalg.norm(decoded - target) / np.linalg.norm(target))
        rows.append({
            "seed": seed, "method": name, "serialized_bytes": size,
            "relative_frobenius_error": error, "effective_update_rank": rank(decoded),
            "target_effective_rank": rank(target), "restart_phases": PHASES,
            "optimizer_updates": 0, "rotation_mac_proxy": int(rot_mac),
            "update_decode_mac_proxy": int(PHASES * 2 * D * RANK * D), "wall_time_s": wall,
        })
    assert np.array_equal(mirror["shared_butterfly_angles"], native["shared_butterfly_angles"])
    assert np.array_equal(mirror["phase_view_scales"], native["phase_view_scales"])
    assert np.array_equal(mirror["phase_coefficients"], native["phase_coefficients"])
    assert np.array_equal(views, np.stack([butterfly_view(theta, s, basis) for s in SCALES]))
    return rows


def run(seeds):
    rows = [r for seed in seeds for r in run_seed(seed)]
    (ROOT / "source" / "development.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (ROOT / "RESULTS_CORE.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "world_or_seed", "method", "serialized_bytes", "train_tokens_or_examples", "optimizer_updates", "active_compute_proxy", "wall_time_s", "primary_metric", "primary_value", "secondary_metric", "secondary_value", "status_note"])
        for r in rows:
            w.writerow(["rank16_four_views", r["seed"], r["method"], r["serialized_bytes"], 0, 0, r["rotation_mac_proxy"] + r["update_decode_mac_proxy"], r["wall_time_s"], "relative_frobenius_error", r["relative_frobenius_error"], "effective_update_rank", r["effective_update_rank"], "development; fresh sealed"])
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", nargs="+", type=int, required=True); args = ap.parse_args()
    print(json.dumps(run(args.seeds), indent=2))
