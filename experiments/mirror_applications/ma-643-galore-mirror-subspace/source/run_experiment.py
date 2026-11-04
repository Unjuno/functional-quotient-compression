import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = 32
RANK = 4
TASKS = 16
N = 64
STEPS = 100
BETA1, BETA2, LR, EPS = .9, .99, .03, 1e-8
SCHEMA = np.frombuffer(b"GALORE-CODEBOOK-STATE-V1", dtype=np.uint8)


def make_world(seed, k):
    rng = np.random.default_rng(seed * 100 + k)
    q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    modes = [q[:, j * RANK:(j + 1) * RANK] for j in range(k)]
    tasks = []
    for t in range(TASKS):
        mode = t % k
        z = rng.normal(size=(N, RANK))
        x = z @ modes[mode].T
        a = rng.normal(0, .15, (RANK, D))
        target = modes[mode] @ a
        y = x @ target
        zt = rng.normal(size=(N, RANK))
        xh = zt @ modes[mode].T
        yh = xh @ target
        tasks.append((mode, x, y, xh, yh))
    return tasks


def grad(x, y, w):
    return (2.0 / len(x)) * x.T @ (x @ w - y)


def left_basis(g):
    u, _, _ = np.linalg.svd(g, full_matrices=False)
    return u[:, :RANK]


def optimize(x, y, p, steps=STEPS):
    w = np.zeros((D, D), dtype=np.float64)
    m = np.zeros((RANK, D), dtype=np.float64)
    v = np.zeros((RANK, D), dtype=np.float64)
    for step in range(1, steps + 1):
        g = grad(x, y, w)
        pg = p.T @ g
        m = BETA1 * m + (1 - BETA1) * pg
        v = BETA2 * v + (1 - BETA2) * (pg * pg)
        mh = m / (1 - BETA1 ** step)
        vh = v / (1 - BETA2 ** step)
        w -= LR * p @ (mh / (np.sqrt(vh) + EPS))
    return w, m, v


def save_payload(seed, k, method, inference, resume):
    ip = ROOT / "source" / f"{seed}_k{k}_{method}_inference.npz"
    rp = ROOT / "source" / f"{seed}_k{k}_{method}_resume.npz"
    np.savez_compressed(ip, **inference)
    np.savez_compressed(rp, **resume)
    return ip.stat().st_size, rp.stat().st_size


def run_one(seed, k):
    t0 = time.perf_counter()
    task_data = make_world(seed, k)
    # Discover a shared dictionary from the initial full gradient of one task per mode.
    dictionary = []
    discovery_svds = 0
    for mode in range(k):
        idx = next(i for i, task in enumerate(task_data) if task[0] == mode)
        _, x, y, _, _ = task_data[idx]
        dictionary.append(left_basis(grad(x, y, np.zeros((D, D)))))
        discovery_svds += 1
    dictionary = np.stack(dictionary).astype(np.float32)
    methods = {m: {"weights": [], "m": [], "v": [], "projectors": [], "codes": [], "losses": []}
               for m in ("full_adam", "galore_independent", "mirror_codebook", "native_codebook_id")}
    method_seconds = {m: 0.0 for m in methods}
    for task_idx, (mode, x, y, xh, yh) in enumerate(task_data):
        # Full Adam, dense matrix moments.
        t = time.perf_counter()
        w = np.zeros((D, D), dtype=np.float64)
        m = np.zeros((D, D), dtype=np.float64)
        v = np.zeros((D, D), dtype=np.float64)
        for step in range(1, STEPS + 1):
            g = grad(x, y, w)
            m = BETA1 * m + (1 - BETA1) * g
            v = BETA2 * v + (1 - BETA2) * (g * g)
            w -= LR * (m / (1 - BETA1 ** step)) / (np.sqrt(v / (1 - BETA2 ** step)) + EPS)
        held = float(np.mean((xh @ w - yh) ** 2))
        methods["full_adam"]["weights"].append(w.astype(np.float32)); methods["full_adam"]["m"].append(m.astype(np.float32)); methods["full_adam"]["v"].append(v.astype(np.float32)); methods["full_adam"]["losses"].append(held)
        method_seconds["full_adam"] += time.perf_counter() - t

        # Standard GaLore: per-task SVD projector, projected Adam moments.
        t = time.perf_counter()
        p = left_basis(grad(x, y, np.zeros((D, D)))).astype(np.float32)
        w, gm, gv = optimize(x, y, p.astype(np.float64))
        held = float(np.mean((xh @ w - yh) ** 2))
        methods["galore_independent"]["weights"].append(w.astype(np.float32)); methods["galore_independent"]["m"].append(gm.astype(np.float32)); methods["galore_independent"]["v"].append(gv.astype(np.float32)); methods["galore_independent"]["projectors"].append(p); methods["galore_independent"]["losses"].append(held)
        method_seconds["galore_independent"] += time.perf_counter() - t

        # Both codebook implementations select the exact task mode in this controlled task.
        p = dictionary[mode]
        for name in ("mirror_codebook", "native_codebook_id"):
            t = time.perf_counter()
            w, cm, cv = optimize(x, y, p.astype(np.float64))
            held = float(np.mean((xh @ w - yh) ** 2))
            methods[name]["weights"].append(w.astype(np.float32)); methods[name]["m"].append(cm.astype(np.float32)); methods[name]["v"].append(cv.astype(np.float32)); methods[name]["codes"].append(mode); methods[name]["losses"].append(held)
            method_seconds[name] += time.perf_counter() - t

    rows = []
    task_ids = np.arange(TASKS, dtype=np.uint8)
    steps_arr = np.full(TASKS, STEPS, dtype=np.uint16)
    for method, state in methods.items():
        weights = np.stack(state["weights"])
        inference = {"task_ids": task_ids, "weights": weights, "schema": SCHEMA}
        resume = {**inference, "first_moments": np.stack(state["m"]), "second_moments": np.stack(state["v"]), "step_counts": steps_arr}
        if method == "galore_independent":
            resume["task_projectors"] = np.stack(state["projectors"])
        elif method in ("mirror_codebook", "native_codebook_id"):
            resume["projector_dictionary"] = dictionary
            resume["task_codes"] = np.asarray(state["codes"], dtype=np.uint8)
        inference_bytes, resume_bytes = save_payload(seed, k, method, inference, resume)
        rows.append({
            "seed": seed, "distinct_gradient_modes": k, "method": method,
            "inference_payload_bytes": inference_bytes, "resume_payload_bytes": resume_bytes,
            "mean_heldout_mse": float(np.mean(state["losses"])), "max_heldout_mse": float(np.max(state["losses"])),
            "task_count": TASKS, "training_examples": TASKS * N,
            "optimizer_updates": TASKS * STEPS,
            "dense_gradient_mac_proxy": TASKS * STEPS * (2 * N * D * D),
            "projection_mac_proxy": 0 if method == "full_adam" else TASKS * STEPS * (4 * RANK * D * D),
            "projector_discovery_svd_count": TASKS if method == "galore_independent" else (discovery_svds if method in ("mirror_codebook", "native_codebook_id") else 0),
            "wall_time_s": method_seconds[method],
        })
    # Functional alias is inherent in the registered ordinary dictionary-ID control.
    for a, b in zip(methods["mirror_codebook"]["weights"], methods["native_codebook_id"]["weights"]):
        assert np.array_equal(a, b)
    return rows


def run(seeds):
    rows = []
    for seed in seeds:
        for k in (1, 2, 4, 8):
            rows.extend(run_one(seed, k))
    (ROOT / "source" / "development.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (ROOT / "RESULTS_CORE.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "world_or_seed", "method", "serialized_bytes", "train_tokens_or_examples", "optimizer_updates", "active_compute_proxy", "wall_time_s", "primary_metric", "primary_value", "secondary_metric", "secondary_value", "status_note"])
        for r in rows:
            w.writerow([f"modes_{r['distinct_gradient_modes']}", r["seed"], r["method"], r["resume_payload_bytes"], r["training_examples"], r["optimizer_updates"], r["dense_gradient_mac_proxy"] + r["projection_mac_proxy"], r["wall_time_s"], "mean_heldout_mse", r["mean_heldout_mse"], "inference_payload_bytes", r["inference_payload_bytes"], "development; fresh sealed"])
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.seeds), indent=2))
