"""MA-312 shared intrinsic basis and high-multiplicity task-code screen."""
from __future__ import annotations

import hashlib
import io
import time
import zipfile
from pathlib import Path

import numpy as np

D, K, T, NTR, NTE = 32, 8, 256, 64, 128
RADIUS = np.float32(0.25)
GRID = np.linspace(-np.pi, np.pi, 720, endpoint=False, dtype=np.float32)
METHODS = ("tied", "said8", "coeff2", "mirror", "independent")


def make_world(seed: int, kind: str):
    rng = np.random.default_rng(seed)
    theta0 = rng.normal(0, 0.15, D).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(D, K)))
    basis = q.astype(np.float32)
    if kind == "aligned":
        angles = rng.uniform(-np.pi, np.pi, T).astype(np.float32)
        z = np.zeros((T, K), dtype=np.float32)
        z[:, 0] = RADIUS * np.cos(angles)
        z[:, 1] = RADIUS * np.sin(angles)
    elif kind == "unrelated":
        angles = np.zeros(T, dtype=np.float32)
        z = rng.normal(0, 0.25, (T, K)).astype(np.float32)
    else:
        raise ValueError(kind)
    targets = theta0[None, :] + z @ basis.T
    xtr = rng.normal(size=(T, NTR, D)).astype(np.float32)
    xte = rng.normal(size=(T, NTE, D)).astype(np.float32)
    ytr = np.einsum("tnd,td->tn", xtr, targets, optimize=True)
    yte = np.einsum("tnd,td->tn", xte, targets, optimize=True)
    return theta0, basis, z, angles, targets, xtr, ytr, xte, yte


def fit_said(xtr, ytr, theta0, basis):
    residual = ytr - np.einsum("tnd,d->tn", xtr, theta0, optimize=True)
    z = np.empty((T, K), dtype=np.float32)
    for t in range(T):
        z[t] = np.linalg.lstsq(xtr[t] @ basis, residual[t], rcond=None)[0]
    return {"theta0": theta0, "basis": basis, "z_f32": z}


def fit_coeff2(xtr, ytr, theta0, basis):
    p2 = basis[:, :2]
    residual = ytr - np.einsum("tnd,d->tn", xtr, theta0, optimize=True)
    coeff = np.empty((T, 2), dtype=np.float32)
    for t in range(T):
        coeff[t] = np.linalg.lstsq(xtr[t] @ p2, residual[t], rcond=None)[0]
    return {"theta0": theta0, "P2": p2, "coeff_f16": coeff.astype(np.float16)}


def fit_mirror(xtr, ytr, theta0, basis):
    p2 = basis[:, :2]
    angles = np.empty(T, dtype=np.float32)
    cos_grid, sin_grid = np.cos(GRID), np.sin(GRID)
    for t in range(T):
        features = xtr[t] @ p2
        residual = ytr[t] - xtr[t] @ theta0
        a, b = RADIUS * features[:, 0], RADIUS * features[:, 1]
        err = np.mean((a[:, None] * cos_grid + b[:, None] * sin_grid - residual[:, None]) ** 2, axis=0)
        angles[t] = GRID[int(np.argmin(err))]
    codes = np.concatenate((np.array([RADIUS], dtype=np.float32), angles)).astype(np.float16)
    return {"theta0": theta0, "P2": p2, "view_codes_f16": codes}


def fit_independent(xtr, ytr):
    weights = np.empty((T, D), dtype=np.float32)
    for t in range(T):
        weights[t] = np.linalg.lstsq(xtr[t], ytr[t], rcond=None)[0]
    return {"weights_f32": weights}


def make_state(method, theta0, basis, xtr, ytr):
    if method == "tied":
        return {"theta0": theta0}
    if method == "said8":
        return fit_said(xtr, ytr, theta0, basis)
    if method == "coeff2":
        return fit_coeff2(xtr, ytr, theta0, basis)
    if method == "mirror":
        return fit_mirror(xtr, ytr, theta0, basis)
    if method == "independent":
        return fit_independent(xtr, ytr)
    raise ValueError(method)


def npy_bytes(array):
    b = io.BytesIO()
    np.lib.format.write_array(b, np.ascontiguousarray(array), allow_pickle=False)
    return b.getvalue()


def save_payload(path: Path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(state):
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            zf.writestr(info, npy_bytes(state[name]))
    return path.stat().st_size


def load_payload(path):
    with zipfile.ZipFile(path) as zf:
        return {name[:-4]: np.load(io.BytesIO(zf.read(name)), allow_pickle=False)
                for name in sorted(zf.namelist())}


def decode_theta(method, state, task):
    if method == "tied":
        return state["theta0"]
    if method == "said8":
        return state["theta0"] + state["basis"] @ state["z_f32"][task]
    if method == "coeff2":
        return state["theta0"] + state["P2"] @ state["coeff_f16"][task].astype(np.float32)
    if method == "mirror":
        radius = float(state["view_codes_f16"][0])
        angle = float(state["view_codes_f16"][task + 1])
        z = np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32)
        return state["theta0"] + state["P2"] @ z
    if method == "independent":
        return state["weights_f32"][task]
    raise ValueError(method)


def fit_ops_proxy(method):
    if method == "mirror":
        return T * NTR * len(GRID) * 6
    if method == "said8":
        return T * (NTR * K * K + K ** 3)
    if method == "coeff2":
        return T * (NTR * 4 + 8)
    if method == "independent":
        return T * (NTR * D * D + D ** 3)
    return 0


def active_ops(method):
    if method == "said8":
        return D * K + D
    if method == "coeff2":
        return D * 2 + D
    if method == "mirror":
        return D * 2 + D + 6
    return D


def evaluate(method, state, x, y):
    errs, norms = [], []
    start = time.perf_counter()
    for repeat in range(3):
        for task in range(T):
            pred = x[task] @ decode_theta(method, state, task)
            if repeat == 0:
                e = float(np.mean((pred - y[task]) ** 2))
                errs.append(e)
                norms.append(e / max(float(np.mean(y[task] ** 2)), 1e-12))
    wall = (time.perf_counter() - start) / 3
    examples = T * x.shape[1]
    return {"mse": float(np.mean(errs)), "normalized_mse": float(np.mean(norms)),
            "test_examples": examples, "support_examples": T * NTR, "optimizer_updates": 0,
            "fit_compute_proxy": fit_ops_proxy(method), "active_ops_per_example": active_ops(method),
            "active_ops_total": active_ops(method) * examples, "wall_time_s": wall,
            "examples_per_s": examples / max(wall, 1e-12)}


def run_world(seed, kind, outdir):
    theta0, basis, z, angles, targets, xtr, ytr, xte, yte = make_world(seed, kind)
    rows = []
    for method in METHODS:
        t0 = time.perf_counter()
        state = make_state(method, theta0, basis, xtr, ytr)
        encode = time.perf_counter() - t0
        path = outdir / f"{kind}_{seed}_{method}.npz"
        nbytes = save_payload(path, state)
        loaded = load_payload(path)
        m = evaluate(method, loaded, xte, yte)
        rows.append({"condition": kind, "world_or_seed": seed, "method": method,
                     "serialized_bytes": nbytes, "parameter_tensor_bytes": sum(v.nbytes for v in state.values()),
                     "support_examples": m["support_examples"], "test_examples": m["test_examples"],
                     "optimizer_updates": 0, "fit_compute_proxy": m["fit_compute_proxy"],
                     "active_ops_per_example": m["active_ops_per_example"],
                     "active_ops_total": m["active_ops_total"], "encode_wall_time_s": encode,
                     "wall_time_s": m["wall_time_s"], "examples_per_s": m["examples_per_s"],
                     "mse": m["mse"], "normalized_mse": m["normalized_mse"],
                     "payload_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows
