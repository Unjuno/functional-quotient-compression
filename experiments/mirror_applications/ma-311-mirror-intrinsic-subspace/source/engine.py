"""MA-311 task-code fitting and deterministic serialized inference harness."""
from __future__ import annotations

import hashlib
import io
import time
import zipfile
from pathlib import Path

import numpy as np

D, K, T, NTR, NTE = 16, 4, 64, 64, 256
RADIUS = np.float32(0.25)
GRID = np.linspace(-np.pi, np.pi, 720, endpoint=False, dtype=np.float32)
METHODS = ("tied", "said4", "coeff2", "mirror", "independent")


def make_world(seed: int, kind: str):
    rng = np.random.default_rng(seed)
    theta0 = rng.normal(0, 0.2, D).astype(np.float32)
    raw = rng.normal(size=(D, K)).astype(np.float64)
    p, _ = np.linalg.qr(raw)
    p = p.astype(np.float32)
    if kind == "aligned":
        angles = np.linspace(-np.pi, np.pi, T, endpoint=False, dtype=np.float32)
        rng.shuffle(angles)
        z = np.zeros((T, K), dtype=np.float32)
        z[:, 0] = RADIUS * np.cos(angles)
        z[:, 1] = RADIUS * np.sin(angles)
    elif kind == "unrelated":
        angles = np.zeros(T, dtype=np.float32)
        z = rng.normal(0, 0.25, (T, K)).astype(np.float32)
    else:
        raise ValueError(kind)
    targets = theta0[None, :] + z @ p.T
    xtr = rng.normal(size=(T, NTR, D)).astype(np.float32)
    xte = rng.normal(size=(T, NTE, D)).astype(np.float32)
    ytr = np.einsum("tnd,td->tn", xtr, targets, optimize=True)
    yte = np.einsum("tnd,td->tn", xte, targets, optimize=True)
    return theta0, p, z, angles, targets, xtr, ytr, xte, yte


def fit_said(xtr, ytr, theta0, p):
    residual = ytr - np.einsum("tnd,d->tn", xtr, theta0, optimize=True)
    z = np.empty((T, K), dtype=np.float32)
    for t in range(T):
        z[t] = np.linalg.lstsq(xtr[t] @ p, residual[t], rcond=None)[0]
    return {"theta0": theta0, "P": p, "z": z}


def fit_coeff2(xtr, ytr, theta0, p):
    p2 = p[:, :2]
    residual = ytr - np.einsum("tnd,d->tn", xtr, theta0, optimize=True)
    coeff = np.empty((T, 2), dtype=np.float32)
    for t in range(T):
        coeff[t] = np.linalg.lstsq(xtr[t] @ p2, residual[t], rcond=None)[0]
    return {"theta0": theta0, "P2": p2, "coeff_f16": coeff.astype(np.float16)}


def fit_mirror(xtr, ytr, theta0, p):
    p2 = p[:, :2]
    angles = np.empty(T, dtype=np.float32)
    for t in range(T):
        feature = xtr[t] @ p2
        residual = ytr[t] - xtr[t] @ theta0
        c = RADIUS * feature[:, 0]
        s = RADIUS * feature[:, 1]
        errors = np.mean((c[:, None] * np.cos(GRID)[None, :] +
                          s[:, None] * np.sin(GRID)[None, :] - residual[:, None]) ** 2, axis=0)
        angles[t] = GRID[int(np.argmin(errors))]
    return {"theta0": theta0, "P2": p2, "radius_f16": np.array([RADIUS], dtype=np.float16),
            "angles_f16": angles.astype(np.float16)}


def fit_independent(xtr, ytr):
    weights = np.empty((T, D), dtype=np.float32)
    for t in range(T):
        weights[t] = np.linalg.lstsq(xtr[t], ytr[t], rcond=None)[0]
    return {"weights": weights}


def make_state(method, theta0, p, xtr, ytr):
    if method == "tied":
        return {"theta0": theta0}
    if method == "said4":
        return fit_said(xtr, ytr, theta0, p)
    if method == "coeff2":
        return fit_coeff2(xtr, ytr, theta0, p)
    if method == "mirror":
        return fit_mirror(xtr, ytr, theta0, p)
    if method == "independent":
        return fit_independent(xtr, ytr)
    raise ValueError(method)


def npy_bytes(array):
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.ascontiguousarray(array), allow_pickle=False)
    return buf.getvalue()


def save_payload(path: Path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(state):
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            zf.writestr(info, npy_bytes(state[name]))
    return path.stat().st_size


def load_payload(path: Path):
    with zipfile.ZipFile(path) as zf:
        return {name[:-4]: np.load(io.BytesIO(zf.read(name)), allow_pickle=False)
                for name in sorted(zf.namelist())}


def decode_theta(method, state, task):
    if method == "tied":
        return state["theta0"]
    if method == "said4":
        return state["theta0"] + state["P"] @ state["z"][task]
    if method == "coeff2":
        return state["theta0"] + state["P2"] @ state["coeff_f16"][task].astype(np.float32)
    if method == "mirror":
        a = float(state["angles_f16"][task])
        r = float(state["radius_f16"][0])
        z = np.array([r * np.cos(a), r * np.sin(a)], dtype=np.float32)
        return state["theta0"] + state["P2"] @ z
    if method == "independent":
        return state["weights"][task]
    raise ValueError(method)


def op_proxy(method):
    if method == "said4":
        return D * K + D
    if method == "coeff2":
        return D * 2 + D
    if method == "mirror":
        return D * 2 + D + 6  # P2 @ (r*cos,sin) + prediction + trig
    return D


def fit_ops_proxy(method):
    if method == "mirror":
        return T * len(GRID) * NTR * 6
    if method == "said4":
        return T * (NTR * K * K + K ** 3)
    if method == "coeff2":
        return T * (NTR * 4 + 8)
    if method == "independent":
        return T * (NTR * D * D + D ** 3)
    return 0


def evaluate(method, state, x, y):
    per_task_mse = []
    per_task_norm = []
    start = time.perf_counter()
    for repeat in range(3):
        for t in range(T):
            pred = x[t] @ decode_theta(method, state, t)
            if repeat == 0:
                err = float(np.mean((pred - y[t]) ** 2))
                denom = float(np.mean(y[t] ** 2))
                per_task_mse.append(err)
                per_task_norm.append(err / max(denom, 1e-12))
    wall = (time.perf_counter() - start) / 3
    examples = T * x.shape[1]
    return {"mse": float(np.mean(per_task_mse)), "normalized_mse": float(np.mean(per_task_norm)),
            "examples": examples, "optimizer_updates": 0, "fit_compute_proxy": fit_ops_proxy(method),
            "active_ops_per_example": op_proxy(method), "active_ops_total": op_proxy(method) * examples,
            "wall_time_s": wall, "examples_per_s": examples / max(wall, 1e-12)}


def run_world(seed, kind, outdir):
    theta0, p, z, angles, targets, xtr, ytr, xte, yte = make_world(seed, kind)
    rows = []
    for method in METHODS:
        t0 = time.perf_counter()
        state = make_state(method, theta0, p, xtr, ytr)
        encode_wall = time.perf_counter() - t0
        path = outdir / f"{kind}_{seed}_{method}.npz"
        byte_count = save_payload(path, state)
        loaded = load_payload(path)
        metrics = evaluate(method, loaded, xte, yte)
        rows.append({"condition": kind, "world_or_seed": seed, "method": method,
                     "serialized_bytes": byte_count,
                     "parameter_tensor_bytes": sum(a.nbytes for a in state.values()),
                     "support_examples": T * NTR, "test_examples": metrics["examples"],
                     "optimizer_updates": 0, "fit_compute_proxy": metrics["fit_compute_proxy"],
                     "active_ops_per_example": metrics["active_ops_per_example"],
                     "active_ops_total": metrics["active_ops_total"], "encode_wall_time_s": encode_wall,
                     "wall_time_s": metrics["wall_time_s"], "examples_per_s": metrics["examples_per_s"],
                     "mse": metrics["mse"], "normalized_mse": metrics["normalized_mse"],
                     "payload_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows
