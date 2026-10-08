"""MA-257 deterministic compositional context mechanism screen."""
from __future__ import annotations

import hashlib
import math
import struct
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

D = 6
K = 8
METHODS = ("shared", "mirror_f16", "mirror_f32", "native_rotation_f16", "coeff_f16", "coeff_f32", "task_angles_f16", "psp_sign", "independent_full")
METHOD_TAGS = {name: i for i, name in enumerate(METHODS)}


def rotation(theta: float) -> np.ndarray:
    c, s = math.cos(float(theta)), math.sin(float(theta))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def apply_plane(z: np.ndarray, plane: int, a: float, b: float) -> np.ndarray:
    """Right multiply one 2x2 scalar/skew block, on the supplied array."""
    out = z.copy()
    i = 2 * plane
    u, v = z[..., i].copy(), z[..., i + 1].copy()
    out[..., i] = a * u + b * v
    out[..., i + 1] = -b * u + a * v
    return out


def task_index(a: int, b: int, c: int) -> int:
    return (a * K + b) * K + c


def factors(idx: int) -> tuple[int, int, int]:
    return idx // (K * K), (idx // K) % K, idx % K


def support_mask(support_residues: int) -> np.ndarray:
    return np.array([sum(v * m for v, m in zip(factors(i), (1, 3, 5))) % K < support_residues for i in range(K**3)])


@dataclass
class World:
    seed: int
    condition: str
    weights: np.ndarray
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


def build_world(seed: int, condition: str) -> World:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.18, size=(D, D)) + 1.2 * np.eye(D)
    angles = np.zeros((3, K), dtype=np.float64)
    angles[:, 1:] = rng.uniform(-0.8, 0.8, size=(3, K - 1))
    weights = np.empty((K**3, D, D), dtype=np.float64)
    for idx in range(K**3):
        a, b, c = factors(idx)
        if condition == "aligned":
            q = np.eye(D)
            for p, level in enumerate((a, b, c)):
                q[2*p:2*p+2, 2*p:2*p+2] = rotation(angles[p, level])
            weights[idx] = base @ q
        elif condition == "independent":
            weights[idx] = base if idx == 0 else rng.normal(0.0, 0.2, size=(D, D))
        else:
            raise ValueError(condition)

    # Each combination has independent examples and fixed train/validation/test partitions.
    xs, ys = [], []
    for w in weights:
        xtr = rng.normal(size=(32, D)); ytr = xtr @ w
        xv = rng.normal(size=(64, D)); yv = xv @ w
        xt = rng.normal(size=(128, D)); yt = xt @ w
        xs.append((xtr, xv, xt)); ys.append((ytr, yv, yt))
    return World(seed, condition, weights,
                 np.stack([x[0] for x in xs]), np.stack([y[0] for y in ys]),
                 np.stack([x[1] for x in xs]), np.stack([y[1] for y in ys]),
                 np.stack([x[2] for x in xs]), np.stack([y[2] for y in ys]))


def estimate_map(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=1e-12)[0]


def normalized_mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2) / (np.mean(target ** 2) + 1e-12))


def extract_factor_codes(base: np.ndarray, estimates: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-factor angle and unrestricted (cos,sin)-pair tables from support maps only."""
    samples: list[list[list[tuple[float, float]]]] = [[[] for _ in range(K)] for _ in range(3)]
    for idx in np.flatnonzero(mask):
        rel = np.linalg.solve(base, estimates[idx])
        for p, level in enumerate(factors(int(idx))):
            block = rel[2*p:2*p+2, 2*p:2*p+2]
            cosine = float((block[0, 0] + block[1, 1]) / 2.0)
            sine = float((block[1, 0] - block[0, 1]) / 2.0)
            samples[p][level].append((cosine, sine))
    coeff = np.zeros((3, K, 2), dtype=np.float64)
    theta = np.zeros((3, K), dtype=np.float64)
    for p in range(3):
        for level in range(K):
            vals = samples[p][level]
            if not vals:
                raise ValueError(f"support split leaves factor {p} level {level} unseen")
            mean = np.mean(np.asarray(vals), axis=0)
            coeff[p, level] = mean
            theta[p, level] = math.atan2(float(mean[1]), float(mean[0]))
    return theta, coeff


def pack_records(records: list[tuple[str, np.ndarray]]) -> bytes:
    out = bytearray(b"MA257\x01")
    out += struct.pack("<I", len(records))
    for name, value in records:
        a = np.ascontiguousarray(value)
        if a.dtype.kind == "f":
            dtype = 1 if a.dtype.itemsize <= 2 else 2
            a = a.astype("<f2" if dtype == 1 else "<f4", copy=False)
        elif a.dtype.kind in "ui":
            dtype = 3
            a = a.astype("u1", copy=False)
        else:
            raise TypeError(a.dtype)
        n = name.encode("utf-8")
        out += struct.pack("<H", len(n)) + n
        out += struct.pack("<BB", dtype, a.ndim)
        if a.ndim:
            out += struct.pack("<" + "I" * a.ndim, *a.shape)
        out += a.tobytes()
    return bytes(out)


def unpack_records(payload: bytes) -> dict[str, np.ndarray]:
    view = memoryview(payload)
    if bytes(view[:6]) != b"MA257\x01":
        raise ValueError("bad MA257 payload header")
    pos = 6
    count = struct.unpack_from("<I", view, pos)[0]; pos += 4
    out: dict[str, np.ndarray] = {}
    for _ in range(count):
        nlen = struct.unpack_from("<H", view, pos)[0]; pos += 2
        name = bytes(view[pos:pos+nlen]).decode("utf-8"); pos += nlen
        dtype, ndim = struct.unpack_from("<BB", view, pos); pos += 2
        shape = struct.unpack_from("<" + "I" * ndim, view, pos) if ndim else (); pos += 4 * ndim
        dt = {1: "<f2", 2: "<f4", 3: "u1"}[dtype]
        nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dt).itemsize
        size = nbytes if ndim else np.dtype(dt).itemsize
        arr = np.frombuffer(view[pos:pos+size], dtype=dt).copy().reshape(shape); pos += size
        out[name] = arr
    if pos != len(view):
        raise ValueError("payload has trailing bytes")
    return out


def psp_context(seed: int, idx: int) -> np.ndarray:
    return np.random.default_rng(int(seed) * 1009 + 17 + int(idx) * 9176).choice(np.array([-1.0, 1.0]), size=(D, D))


def _meta(method: str) -> list[tuple[str, np.ndarray]]:
    return [("method_id", np.array([METHOD_TAGS[method]], dtype=np.uint8)),
            ("schema", np.frombuffer(b"D=6;K=8;three-disjoint-givens;v1", dtype=np.uint8))]


def build_payload(method: str, base: np.ndarray, theta: np.ndarray, coeff: np.ndarray,
                  task_theta: np.ndarray, estimates: np.ndarray, all_weights: np.ndarray,
                  support: np.ndarray, seed: int) -> bytes:
    r = _meta(method)
    if method == "shared":
        r += [("W", base)]
    elif method.startswith("mirror_") or method == "native_rotation_f16":
        r += [("W", base), ("theta", theta.astype(np.float16 if method.endswith("f16") else np.float32))]
    elif method.startswith("coeff_"):
        r += [("W", base), ("coeff", coeff.astype(np.float16 if method.endswith("f16") else np.float32))]
    elif method == "task_angles_f16":
        r += [("W", base), ("support", support.astype(np.uint8)), ("angles", task_theta.astype(np.float16))]
    elif method == "psp_sign":
        # Deterministically reconstructible contexts are generated from a paid seed and task index.
        sign_seed = int(seed * 1009 + 17)
        signs = np.stack([psp_context(seed, i) for i in range(K**3)])
        superposed = np.sum(all_weights * signs, axis=0)
        r += [("superposed_W", superposed), ("context_seed", np.array([sign_seed], dtype=np.uint32))]
    elif method == "independent_full":
        r += [("task_W", all_weights)]
    else:
        raise ValueError(method)
    return pack_records(r)


def predict(payload_state: dict[str, np.ndarray], method: str, idx: int, x: np.ndarray) -> np.ndarray:
    if method == "shared":
        return x @ payload_state["W"]
    if method.startswith("mirror_") or method == "native_rotation_f16":
        z = x @ payload_state["W"]
        a, b, c = factors(idx)
        for p, level in enumerate((a, b, c)):
            angle = float(payload_state["theta"][p, level])
            z = apply_plane(z, p, math.cos(angle), math.sin(angle))
        return z
    if method.startswith("coeff_"):
        z = x @ payload_state["W"]
        a, b, c = factors(idx)
        for p, level in enumerate((a, b, c)):
            ca, sb = map(float, payload_state["coeff"][p, level])
            z = apply_plane(z, p, ca, sb)
        return z
    if method == "task_angles_f16":
        if not bool(payload_state["support"][idx]):
            raise KeyError("per-task angle control has no held-out composition code")
        z = x @ payload_state["W"]
        for p, angle in enumerate(payload_state["angles"][idx]):
            z = apply_plane(z, p, math.cos(float(angle)), math.sin(float(angle)))
        return z
    if method == "psp_sign":
        sign_seed = int(payload_state["context_seed"][0])
        signs = psp_context(sign_seed, idx)
        return x @ (payload_state["superposed_W"] * signs)
    if method == "independent_full":
        return x @ payload_state["task_W"][idx]
    raise ValueError(method)


def _metrics(world: World, method: str, state: dict[str, np.ndarray], support: np.ndarray) -> dict[str, Any]:
    seen, held, val_seen, val_held = [], [], [], []
    for idx in range(K**3):
        try:
            pred = predict(state, method, idx, world.test_x[idx])
            vpred = predict(state, method, idx, world.val_x[idx])
        except KeyError:
            continue
        score = normalized_mse(pred, world.test_y[idx])
        vscore = normalized_mse(vpred, world.val_y[idx])
        (seen if support[idx] else held).append(score)
        (val_seen if support[idx] else val_held).append(vscore)
    def summary(xs: list[float]) -> tuple[float | None, float | None]:
        return (float(np.mean(xs)), float(np.max(xs))) if xs else (None, None)
    ms, xs = summary(seen), summary(held)
    all_scores = seen + held
    ma = summary(all_scores)
    vms, vxs = summary(val_seen), summary(val_held)
    vma = summary(val_seen + val_held)
    return {"mean_val_seen_nMSE": vms[0], "max_val_seen_nMSE": vms[1],
            "mean_val_heldout_nMSE": vxs[0], "max_val_heldout_nMSE": vxs[1],
            "mean_val_all_nMSE": vma[0], "max_val_all_nMSE": vma[1],
            "mean_seen_nMSE": ms[0], "max_seen_nMSE": ms[1],
            "mean_heldout_nMSE": xs[0], "max_heldout_nMSE": xs[1],
            "mean_all_nMSE": ma[0], "max_all_nMSE": ma[1],
            "evaluated_tasks": len(all_scores)}


def _timing(world: World, method: str, state: dict[str, np.ndarray], support: np.ndarray) -> float:
    # One complete scan, twice, over the actual serialized-state path.
    n = 0
    start = time.perf_counter()
    for _ in range(2):
        for idx in range(K**3):
            if method == "task_angles_f16" and not support[idx]:
                continue
            predict(state, method, idx, world.test_x[idx])
            n += len(world.test_x[idx])
    elapsed = time.perf_counter() - start
    return float(n / max(elapsed, 1e-12))


def fit_and_measure(world: World, support_residues: int) -> list[dict[str, Any]]:
    support = support_mask(support_residues)
    assert support[0]
    # Validate that all factor values are seen by support before fitting.
    for p in range(3):
        assert set(factors(int(i))[p] for i in np.flatnonzero(support)) == set(range(K))
    start = time.perf_counter()
    estimates = np.empty_like(world.weights)
    for idx in np.flatnonzero(support):
        estimates[idx] = estimate_map(world.train_x[idx], world.train_y[idx])
    base = estimates[0]
    theta, coeff = extract_factor_codes(base, estimates, support)
    # Diagnostics for a per-combination angle table are available only on support combinations.
    task_theta = np.zeros((K**3, 3), dtype=np.float64)
    for idx in np.flatnonzero(support):
        rel = np.linalg.solve(base, estimates[idx])
        for p in range(3):
            block = rel[2*p:2*p+2, 2*p:2*p+2]
            cc = (block[0, 0] + block[1, 1]) / 2.0
            ss = (block[1, 0] - block[0, 1]) / 2.0
            task_theta[idx, p] = math.atan2(float(ss), float(cc))
    fit_elapsed = time.perf_counter() - start

    rows = []
    for method in METHODS:
        method_start = time.perf_counter()
        payload = build_payload(method, base, theta, coeff, task_theta, estimates, world.weights, support, world.seed)
        state = unpack_records(payload)
        fit_s = fit_elapsed
        if method == "independent_full":
            # Independent upper gets direct 32-example calibration on every task, including held-out combinations.
            full_start = time.perf_counter()
            full = np.stack([estimate_map(world.train_x[i], world.train_y[i]) for i in range(K**3)])
            payload = build_payload(method, base, theta, coeff, task_theta, estimates, full, np.ones(K**3, bool), world.seed)
            state = unpack_records(payload)
            fit_s = full_start and time.perf_counter() - full_start
        elif method == "task_angles_f16":
            pass
        elif method == "psp_sign":
            # Include actual construction work, not just byte serialization.
            fit_s = fit_elapsed + (time.perf_counter() - method_start)
        metric = _metrics(world, method, state, support)
        fit_proxy = int(np.count_nonzero(support) * (2 * 32 * D * D + D**3) + 3 * K * D * D)
        if method == "independent_full":
            fit_proxy = int(K**3 * (2 * 32 * D * D + D**3))
        elif method == "psp_sign":
            fit_proxy += K**3 * D * D
        # Runtime always executes from the deserialized exact payload.
        rate = _timing(world, method, state, support)
        rows.append({
            "condition": world.condition, "seed": world.seed, "method": method,
            "support_residues": support_residues, "support_combinations": int(support.sum()),
            "serialized_bytes": len(payload), "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "train_examples": int(support.sum() * 32 if method != "independent_full" else K**3 * 32),
            "validation_examples": K**3 * 64, "test_examples": K**3 * 128,
            "source_task_maps_accessed": K**3 if method == "psp_sign" else 0,
            "optimizer_updates": 0, "fit_compute_proxy": fit_proxy, "fit_wall_seconds": float(fit_s),
            "inference_examples_per_second": rate,
            "operator_workspace_bytes": (14336 if method.startswith("mirror_") or method.startswith("coeff_") or method in ("native_rotation_f16", "task_angles_f16") else (6528 if method == "psp_sign" else 6144)),
            "raw_tensor_bytes": int(sum(v.nbytes for v in state.values())),
            **metric,
            "notes": "independent_full uses direct per-task calibration on held-out combinations; PSP uses the full generated task bank; other compositional methods fit support combinations only"
        })
    return rows


def run_world(seed: int, condition: str, support_residues: int) -> list[dict[str, Any]]:
    return fit_and_measure(build_world(seed, condition), support_residues)
