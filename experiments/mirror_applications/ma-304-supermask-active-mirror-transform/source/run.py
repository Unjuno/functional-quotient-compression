import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np

N = 32
K = 256
N_ALIGNED = 128
N_PRIVATE = 16
N_TASKS = N_ALIGNED + N_PRIVATE
RHO = 1.0
GRID = 512
THRESHOLD = 0.02
METHODS = ["dense_shared", "supmask_binary", "hard_tied_sparse", "gate_scalar", "direct_pair_private", "mirror_phase_private", "independent_full"]


def sparse_matrix(indices, values):
    out = np.zeros((N, N), np.float32)
    out.reshape(-1)[indices.astype(int)] = values.astype(np.float32)
    return out


def make_world(seed):
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(N * N, K, replace=False)).astype(np.uint16)
    b1v = rng.normal(0, 0.15, K).astype(np.float32)
    b2v = rng.normal(0, 0.15, K).astype(np.float32)
    b1, b2 = sparse_matrix(indices, b1v), sparse_matrix(indices, b2v)
    dense = rng.normal(0, 0.08, (N, N)).astype(np.float32)
    angles = rng.uniform(-np.pi, np.pi, N_ALIGNED).astype(np.float32)
    targets = []
    groups = []
    for i in range(N_TASKS):
        if i < N_ALIGNED:
            w = np.cos(angles[i]) * b1 + np.sin(angles[i]) * b2
            groups.append("aligned_orbit")
        else:
            idx = np.sort(rng.choice(N * N, K, replace=False)).astype(np.uint16)
            vals = rng.normal(0, 0.15, K).astype(np.float32)
            w = sparse_matrix(idx, vals)
            groups.append("unrelated")
        targets.append(w)
    xs = []
    ys = []
    for i, w in enumerate(targets):
        row = []
        for count, salt in [(64, 101), (32, 211), (64, 307)]:
            x = np.random.default_rng(seed + salt + i * 73).normal(size=(count, N)).astype(np.float32)
            row.append((x, x @ w.T))
        xs.append(row)
        ys.append(w)
    return {"seed": seed, "indices": indices, "b1v": b1v, "b2v": b2v, "b1": b1, "b2": b2,
            "dense": dense, "angles": angles, "weights": np.stack(targets), "sets": xs,
            "groups": np.asarray(groups)}


def nrmse(pred, target):
    return float(np.mean((pred - target) ** 2) / (np.mean(target ** 2) + 1e-12))


def fit_pair(x, y, b1, b2):
    f1 = x @ b1.T
    f2 = x @ b2.T
    design = np.concatenate([f1.reshape(-1, 1), f2.reshape(-1, 1)], axis=1)
    return np.linalg.lstsq(design, y.reshape(-1), rcond=None)[0].astype(np.float32)


def fit_phase(x, y, b1, b2):
    grid = np.linspace(-np.pi, np.pi, GRID, endpoint=False, dtype=np.float32)
    f1, f2 = x @ b1.T, x @ b2.T
    pred = np.cos(grid)[None, None, :] * f1[:, :, None] + np.sin(grid)[None, None, :] * f2[:, :, None]
    err = np.mean((pred - y[:, :, None]) ** 2, axis=(0, 1))
    j = int(np.argmin(err))
    return float(grid[j]), GRID * len(x) * N * 3


def pack(arrays, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            z.writestr(info, buf.getvalue())
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def unpack(path):
    with zipfile.ZipFile(path) as z:
        return {n[:-4]: np.load(io.BytesIO(z.read(n)), allow_pickle=False) for n in sorted(z.namelist())}


def encode(method, w):
    start = time.perf_counter()
    arrays = {"task_ids": np.arange(N_TASKS, dtype=np.uint16), "metadata": np.asarray([N, K, N_ALIGNED, N_PRIVATE], np.uint16)}
    fitops = 0
    if method == "dense_shared":
        arrays["dense"] = w["dense"].astype("<f2")
    elif method == "supmask_binary":
        arrays["dense"] = w["dense"].astype("<f2")
        masks = []
        for i in range(N_TASKS):
            x, y = w["sets"][i][0]
            corr = np.abs((x.T @ y) * w["dense"]).reshape(-1)
            ids = np.argpartition(corr, -K)[-K:]
            masks.append(np.sort(ids).astype(np.uint16))
        arrays["mask_indices"] = np.stack(masks)
        fitops = N_TASKS * len(w["sets"][0][0][0]) * N * N
    elif method == "hard_tied_sparse":
        arrays["indices"] = w["indices"]
        arrays["values"] = w["b1v"].astype("<f2")
    elif method in ("gate_scalar", "direct_pair_private", "mirror_phase_private"):
        arrays.update({"indices": w["indices"], "b1": w["b1v"].astype("<f2"), "b2": w["b2v"].astype("<f2")})
        code_ids, codes, private_ids, private_idx, private_vals = [], [], [], [], []
        for i, sets in enumerate(w["sets"]):
            x, y = sets[0]
            xv, yv = sets[1]
            if method == "gate_scalar":
                b = w["b1"]
                c = np.sum((x @ b.T) * y) / (np.sum((x @ b.T) ** 2) + 1e-12)
                wc = float(c) * b
                code = [c]
                fitops += len(x) * N * K
            elif method == "direct_pair_private":
                c = fit_pair(x, y, w["b1"], w["b2"])
                wc = c[0] * w["b1"] + c[1] * w["b2"]
                code = c
                fitops += len(x) * N * 2 * K
            else:
                a, ops = fit_phase(x, y, w["b1"], w["b2"])
                wc = np.cos(a) * w["b1"] + np.sin(a) * w["b2"]
                code = [a]
                fitops += ops
            score = nrmse(xv @ wc.T, yv)
            if score > THRESHOLD:
                private_ids.append(i)
                idx = np.flatnonzero(w["weights"][i].reshape(-1)).astype(np.uint16)
                private_idx.append(idx)
                private_vals.append(w["weights"][i].reshape(-1)[idx].astype("<f2"))
            else:
                code_ids.append(i)
                codes.append(code)
        arrays["code_ids"] = np.asarray(code_ids, np.uint16)
        arrays["codes"] = np.asarray(codes, dtype="<f2").reshape(-1, 2 if method == "direct_pair_private" else 1)
        arrays["private_ids"] = np.asarray(private_ids, np.uint16)
        arrays["private_indices"] = np.asarray(private_idx, dtype=np.uint16).reshape(-1, K)
        arrays["private_values"] = np.asarray(private_vals, dtype="<f2").reshape(-1, K)
        if method == "mirror_phase_private":
            arrays["radius"] = np.asarray([RHO], dtype="<f2")
    elif method == "independent_full":
        arrays["weights"] = w["weights"].astype("<f2")
    return arrays, fitops, time.perf_counter() - start


def decode(method, s, task, w):
    if method == "dense_shared":
        return s["dense"].astype(np.float32)
    if method == "supmask_binary":
        out = np.zeros((N, N), np.float32)
        out.reshape(-1)[s["mask_indices"][task].astype(int)] = s["dense"].reshape(-1)[s["mask_indices"][task].astype(int)].astype(np.float32)
        return out
    if method == "hard_tied_sparse":
        return sparse_matrix(s["indices"], s["values"])
    if method == "independent_full":
        return s["weights"][task].astype(np.float32)
    if task in s["private_ids"].astype(int):
        j = int(np.where(s["private_ids"] == task)[0][0])
        return sparse_matrix(s["private_indices"][j], s["private_values"][j])
    j = int(np.where(s["code_ids"] == task)[0][0])
    if method == "gate_scalar":
        return float(s["codes"][j, 0]) * sparse_matrix(s["indices"], s["b1"])
    if method == "direct_pair_private":
        a, b = s["codes"][j].astype(np.float32)
        return a * sparse_matrix(s["indices"], s["b1"]) + b * sparse_matrix(s["indices"], s["b2"])
    a = float(s["codes"][j, 0])
    radius = float(s["radius"][0])
    return radius * (np.cos(a) * sparse_matrix(s["indices"], s["b1"]) + np.sin(a) * sparse_matrix(s["indices"], s["b2"]))


def run(seed, split, outdir, jsonpath):
    w = make_world(seed)
    summaries = []
    for method in METHODS:
        arrays, fitops, fitwall = encode(method, w)
        path = Path(outdir) / f"{split}_{seed}_{method}.npz"
        nbytes, digest = pack(arrays, path)
        state = unpack(path)
        scores, aligned, unrelated, active = [], [], [], []
        start = time.perf_counter()
        for i, sets in enumerate(w["sets"]):
            x, y = sets[2]
            pred = x @ decode(method, state, i, w).T
            score = nrmse(pred, y)
            scores.append(score)
            (aligned if i < N_ALIGNED else unrelated).append(score)
            active.append(int(np.count_nonzero(np.abs(decode(method, state, i, w)) > 0)))
        wall = time.perf_counter() - start
        summaries.append({"condition": split, "seed": seed, "method": method, "serialized_bytes": nbytes,
                          "payload_sha256": digest, "support_examples": N_TASKS * 64,
                          "validation_examples": N_TASKS * 32, "test_examples": N_TASKS * 64,
                          "optimizer_updates": 0, "fit_compute_proxy": fitops,
                          "active_edges_per_request": float(np.mean(active)),
                          "aligned_max_n_mse": float(np.max(aligned)), "aligned_mean_n_mse": float(np.mean(aligned)),
                          "unrelated_max_n_mse": float(np.max(unrelated)), "unrelated_mean_n_mse": float(np.mean(unrelated)),
                          "mean_test_n_mse": float(np.mean(scores)), "max_test_n_mse": float(np.max(scores)),
                          "private_tasks": int(len(state.get("private_ids", []))), "fit_wall_s": fitwall,
                          "decode_infer_wall_s": wall, "requests_per_s": N_TASKS / wall})
    result = {"seed": seed, "condition": split, "summaries": summaries}
    p = Path(jsonpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", choices=["development", "fresh"], required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    run(a.seed, a.split, a.outdir, a.json)
