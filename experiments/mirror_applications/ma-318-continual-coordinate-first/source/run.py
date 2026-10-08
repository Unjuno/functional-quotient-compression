import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np

D = 64
TASKS = 12
THRESHOLD = 1e-4
GRID = 1024
METHODS = ["hard_tied", "direct_growing_basis", "mirror_coordinate_first", "independent_full"]
GROUPS = ["initial_orbit"] * 4 + ["second_orbit"] * 4 + ["unrelated"] * 4


def orthonormal_columns(rng):
    q, _ = np.linalg.qr(rng.normal(size=(D, 4)))
    return q.T.astype(np.float32)


def make_world(seed):
    rng = np.random.default_rng(seed)
    a, b, c, d = orthonormal_columns(rng)
    angles = np.concatenate([rng.uniform(-np.pi, np.pi, 4), rng.uniform(-np.pi, np.pi, 4)]).astype(np.float32)
    weights = []
    for i in range(TASKS):
        if i < 4:
            w = np.cos(angles[i]) * a + np.sin(angles[i]) * b
        elif i < 8:
            j = i - 4
            w = np.cos(angles[4 + j]) * c + np.sin(angles[4 + j]) * d
        else:
            w = rng.normal(size=D).astype(np.float32)
            w /= np.linalg.norm(w)
        weights.append(w.astype(np.float32))
    sets = []
    for i, w in enumerate(weights):
        item = []
        for n, salt in [(128, 101), (64, 211), (128, 307)]:
            x = np.random.default_rng(seed + salt + i * 41).normal(size=(n, D)).astype(np.float32)
            item.append((x, x @ w))
        sets.append(item)
    return {"seed": seed, "initial_basis": np.stack([a, b]), "weights": np.stack(weights),
            "angles": angles, "sets": sets, "groups": GROUPS}


def nrmse(pred, target):
    return float(np.mean((pred - target) ** 2) / (np.mean(target ** 2) + 1e-12))


def estimate_weight(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0].astype(np.float32)


def fit_coefficients(x, y, basis):
    features = x @ basis.T
    return np.linalg.lstsq(features, y, rcond=None)[0].astype(np.float32)


def fit_phase(x, y, basis):
    grid = np.linspace(-np.pi, np.pi, GRID, endpoint=False, dtype=np.float32)
    f0 = x @ basis[0]
    f1 = x @ basis[1]
    pred = np.cos(grid)[None, :] * f0[:, None] + np.sin(grid)[None, :] * f1[:, None]
    err = np.mean((pred - y[:, None]) ** 2, axis=0)
    j = int(np.argmin(err))
    return float(grid[j]), GRID * len(x) * D * 2


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


def decode_state(method, state, task):
    if method == "hard_tied":
        return state["weight"].astype(np.float32)
    if method == "independent_full":
        return state["weights"][task].astype(np.float32)
    basis = state["basis"].astype(np.float32)
    if method == "mirror_coordinate_first" and state["modes"][task] == 1:
        phase = float(state["phases"][task])
        return np.cos(phase) * basis[0] + np.sin(phase) * basis[1]
    return state["codes"][task, :len(basis)].astype(np.float32) @ basis


def state_arrays(method, basis, codes, modes, phases, weights, growth):
    common = {"task_ids": np.arange(len(modes), dtype=np.uint8),
              "growth_events": np.asarray(growth, dtype=np.uint8).reshape(-1, 2)}
    if method == "hard_tied":
        return {"weight": weights[0].astype("<f2"), **common}
    if method == "independent_full":
        return {"weights": np.asarray(weights, dtype="<f2"), **common}
    rank = len(basis)
    padded = np.zeros((len(codes), rank), dtype=np.float32)
    for i, code in enumerate(codes):
        padded[i, :min(rank, len(code))] = code[:rank]
    arrays = {"basis": np.asarray(basis, dtype="<f2"), "codes": padded.astype("<f2"), **common}
    if method == "mirror_coordinate_first":
        arrays["modes"] = np.asarray(modes, dtype=np.uint8)
        arrays["phases"] = np.asarray(phases, dtype="<f2")
    return arrays


def run_method(method, w, split, outdir):
    basis = [x.copy() for x in w["initial_basis"]]
    codes, modes, phases, growth = [], [], [], []
    summaries, payload_rows = [], []
    ops_total = 0
    fit_time = 0.0
    for task in range(TASKS):
        start = time.perf_counter()
        x, y = w["sets"][task][0]
        xv, yv = w["sets"][task][1]
        if method == "hard_tied":
            pass
        elif method == "independent_full":
            pass
        else:
            b = np.stack(basis)
            mode, phase, code = 0, 0.0, None
            if method == "mirror_coordinate_first":
                phase, ops = fit_phase(x, y, b[:2])
                ops_total += ops
                pred = np.cos(phase) * (xv @ b[0]) + np.sin(phase) * (xv @ b[1])
                score = nrmse(pred, yv)
                if score <= THRESHOLD:
                    mode = 1
                else:
                    coeff = fit_coefficients(x, y, b)
                    pred = xv @ (coeff @ b)
                    score = nrmse(pred, yv)
                    ops_total += len(x) * D * len(b)
                    if score > THRESHOLD:
                        what = estimate_weight(x, y)
                        residual = what - coeff @ b
                        residual /= np.linalg.norm(residual) + 1e-12
                        basis.append(residual)
                        growth.append([task, len(basis)])
                        b = np.stack(basis)
                        coeff = fit_coefficients(x, y, b)
                        ops_total += len(x) * D * len(b)
                    code = coeff
            else:
                coeff = fit_coefficients(x, y, b)
                pred = xv @ (coeff @ b)
                score = nrmse(pred, yv)
                ops_total += len(x) * D * len(b)
                if score > THRESHOLD:
                    what = estimate_weight(x, y)
                    residual = what - coeff @ b
                    residual /= np.linalg.norm(residual) + 1e-12
                    basis.append(residual)
                    growth.append([task, len(basis)])
                    b = np.stack(basis)
                    coeff = fit_coefficients(x, y, b)
                    ops_total += len(x) * D * len(b)
                code = coeff
            if method == "mirror_coordinate_first":
                modes.append(mode)
                phases.append(phase)
                codes.append(np.zeros(1, np.float32) if mode == 1 else code)
            else:
                codes.append(code)
        fit_time += time.perf_counter() - start
        learned = task + 1
        if method == "hard_tied":
            current_basis, current_codes, current_modes, current_phases = basis, [], [], []
        elif method == "independent_full":
            current_basis, current_codes, current_modes, current_phases = basis, [], [], []
        else:
            current_basis, current_codes, current_modes, current_phases = basis, codes, modes, phases
        arrays = state_arrays(method, current_basis, current_codes, current_modes, current_phases,
                              w["weights"][:learned], growth)
        fpath = Path(outdir) / f"{split}_{w['seed']}_{method}_step{learned:02d}.npz"
        nbytes, digest = pack(arrays, fpath)
        state = unpack(fpath)
        if method == "independent_full":
            # Prefix the paid independent vectors; no future weights are serialized.
            state["weights"] = state["weights"][:learned]
        scores = []
        for old in range(learned):
            xt, yt = w["sets"][old][2]
            pred = xt @ decode_state(method, state, old).astype(np.float32)
            scores.append(nrmse(pred, yt))
        summaries.append({"condition": split, "seed": w["seed"], "method": method, "stream_step": learned,
                          "new_task_group": w["groups"][task], "serialized_bytes": nbytes,
                          "payload_sha256": digest, "basis_rank": len(basis) if method in ("direct_growing_basis", "mirror_coordinate_first") else 0,
                          "basis_growth_count": len(growth), "optimizer_updates": 0,
                          "support_examples": learned * 128, "validation_examples": learned * 64,
                          "test_examples": learned * 128, "fit_compute_proxy_cumulative": ops_total,
                          "fit_wall_s_cumulative": fit_time, "max_prior_test_n_mse": float(np.max(scores)),
                          "mean_prior_test_n_mse": float(np.mean(scores)), "prior_tasks_above_threshold": int(sum(s > THRESHOLD for s in scores))})
        payload_rows.append({"step": learned, "path": str(fpath), "sha256": digest, "bytes": nbytes})
    return summaries, payload_rows


def run(seed, split, outdir, jsonpath):
    w = make_world(seed)
    summaries, packages = [], {}
    for method in METHODS:
        rows, files = run_method(method, w, split, outdir)
        summaries.extend(rows)
        packages[method] = files
    result = {"condition": split, "seed": seed, "summaries": summaries, "packages": packages}
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
