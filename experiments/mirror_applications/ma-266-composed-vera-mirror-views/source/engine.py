"""MA-266 factor-composition screen with VeRA and coefficient controls."""
from __future__ import annotations

import hashlib
import io
import time
import zipfile
from pathlib import Path

import numpy as np

D = 12
F = 4
N = 128
RANK = 6
ALPHA = np.array([-0.60, -0.20, 0.20, 0.60], dtype=np.float32)
BETA = np.array([-0.45, -0.15, 0.15, 0.45], dtype=np.float32)
METHODS = ("tied", "vera_sum", "coeff_product", "mirror", "independent")


def left_rotate(w: np.ndarray, angle: float) -> np.ndarray:
    out = np.array(w, dtype=np.float32, copy=True)
    c, s = np.float32(np.cos(angle)), np.float32(np.sin(angle))
    for k in range(0, D, 2):
        u, v = out[k].copy(), out[k + 1].copy()
        out[k], out[k + 1] = c * u - s * v, s * u + c * v
    return out


def right_rotate(w: np.ndarray, angle: float) -> np.ndarray:
    out = np.array(w, dtype=np.float32, copy=True)
    c, s = np.float32(np.cos(angle)), np.float32(np.sin(angle))
    for k in range(0, D, 2):
        u, v = out[:, k].copy(), out[:, k + 1].copy()
        out[:, k], out[:, k + 1] = c * u - s * v, s * u + c * v
    return out


def make_world(seed: int, kind: str):
    rng = np.random.default_rng(seed)
    core = rng.normal(0.0, 1 / np.sqrt(D), (D, D)).astype(np.float32)
    teachers = np.empty((F, F, D, D), dtype=np.float32)
    for i, a in enumerate(ALPHA):
        for j, b in enumerate(BETA):
            if kind == "aligned":
                teachers[i, j] = right_rotate(left_rotate(core, float(a)), float(b))
            elif kind == "unrelated":
                delta = rng.normal(0.0, 0.40 / np.sqrt(D), (D, D)).astype(np.float32)
                teachers[i, j] = core + delta
            else:
                raise ValueError(kind)
    xs = rng.normal(size=(F, F, N, D)).astype(np.float32)
    ys = np.einsum("ijnd,ijod->ijno", xs, teachers, optimize=True)
    support = np.fromfunction(lambda i, j: ((i + j) % 2 == 0), (F, F), dtype=int).astype(bool)
    return core, teachers, xs, ys, support


def make_vera_state(core: np.ndarray, teachers: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 910000)
    b = rng.normal(0.0, 1 / np.sqrt(D), (D, RANK)).astype(np.float32)
    a = rng.normal(0.0, 1 / np.sqrt(D), (RANK, D)).astype(np.float32)
    basis = np.einsum("dr,rs->rds", b, a, optimize=True).reshape(RANK, -1)
    pairs = [(i, j) for i in range(F) for j in range(F) if (i + j) % 2 == 0]
    # Linear sum of factor-specific VeRA-like scaling coordinates.
    design = np.zeros((len(pairs) * D * D, 2 * F * RANK), dtype=np.float64)
    target = np.empty((len(pairs) * D * D,), dtype=np.float64)
    for p, (i, j) in enumerate(pairs):
        row = slice(p * D * D, (p + 1) * D * D)
        delta = (teachers[i, j] - core).reshape(-1)
        target[row] = delta
        for r in range(RANK):
            vec = basis[r]
            design[row, i * RANK + r] = vec
            design[row, F * RANK + j * RANK + r] = vec
    coeff = np.linalg.lstsq(design, target, rcond=None)[0].reshape(2, F, RANK).astype(np.float32)
    return {"core": core.astype(np.float32), "B": b, "A": a, "factor_codes": coeff}


def make_state(method: str, core: np.ndarray, teachers: np.ndarray, seed: int):
    if method == "tied":
        return {"core": core.astype(np.float32)}
    if method == "vera_sum":
        return make_vera_state(core, teachers, seed)
    if method == "coeff_product":
        # Store ordinary 2D coefficient vectors rather than scalar angle codes.
        return {
            "core": core.astype(np.float32),
            "alpha_cs": np.stack((np.cos(ALPHA), np.sin(ALPHA)), axis=1).astype(np.float16),
            "beta_cs": np.stack((np.cos(BETA), np.sin(BETA)), axis=1).astype(np.float16),
        }
    if method == "mirror":
        return {"core": core.astype(np.float32), "alpha": ALPHA.astype(np.float16), "beta": BETA.astype(np.float16)}
    if method == "independent":
        return {"weights": teachers.astype(np.float32)}
    raise ValueError(method)


def npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.ascontiguousarray(array), allow_pickle=False)
    return stream.getvalue()


def save_payload(path: Path, arrays: dict[str, np.ndarray]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            zf.writestr(info, npy_bytes(arrays[name]))
    return path.stat().st_size


def load_payload(path: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(path) as zf:
        return {name[:-4]: np.load(io.BytesIO(zf.read(name)), allow_pickle=False)
                for name in sorted(zf.namelist())}


def decode(method: str, state: dict[str, np.ndarray], i: int, j: int) -> np.ndarray:
    if method == "tied":
        return state["core"]
    if method == "independent":
        return state["weights"][i, j]
    if method == "mirror":
        return right_rotate(left_rotate(state["core"], float(state["alpha"][i])), float(state["beta"][j]))
    if method == "coeff_product":
        ca, sa = state["alpha_cs"][i].astype(np.float32)
        cb, sb = state["beta_cs"][j].astype(np.float32)
        core = state["core"]
        jl = left_rotate(core, np.pi / 2)
        jr = right_rotate(core, np.pi / 2)
        both = right_rotate(jl, np.pi / 2)
        return (ca * cb * core + sa * cb * jl + ca * sb * jr + sa * sb * both).astype(np.float32)
    if method == "vera_sum":
        b, a = state["B"], state["A"]
        codes = state["factor_codes"][0, i] + state["factor_codes"][1, j]
        delta = (b * codes[None, :]) @ a
        return state["core"] + delta
    raise ValueError(method)


def active_ops(method: str) -> int:
    base = D * D
    if method == "mirror":
        return base + 2 * D * D
    if method == "coeff_product":
        return 5 * base + 12
    if method == "vera_sum":
        return base + 2 * D * RANK
    return base


def evaluate(method: str, state: dict[str, np.ndarray], xs: np.ndarray, ys: np.ndarray,
             pairs: list[tuple[int, int]]) -> dict[str, float | int]:
    err, norm = [], []
    start = time.perf_counter()
    for repeat in range(3):
        for i, j in pairs:
            pred = xs[i, j] @ decode(method, state, i, j).T
            if repeat == 0:
                err.append(float(np.mean((pred - ys[i, j]) ** 2)))
                norm.append(float(np.mean((pred - ys[i, j]) ** 2)) / max(float(np.mean(ys[i, j] ** 2)), 1e-12))
    elapsed = (time.perf_counter() - start) / 3
    examples = len(pairs) * N
    return {"mse": float(np.mean(err)), "normalized_mse": float(np.mean(norm)),
            "examples": examples, "optimizer_updates": 0,
            "active_ops_per_example": active_ops(method),
            "active_ops_total": active_ops(method) * examples,
            "wall_time_s": elapsed, "examples_per_s": examples / max(elapsed, 1e-12)}


def run_world(seed: int, kind: str, output_dir: Path):
    core, teachers, xs, ys, support = make_world(seed, kind)
    rows = []
    for method in METHODS:
        t0 = time.perf_counter()
        state = make_state(method, core, teachers, seed)
        encode_wall = time.perf_counter() - t0
        payload = output_dir / f"{kind}_{seed}_{method}.npz"
        nbytes = save_payload(payload, state)
        loaded = load_payload(payload)
        for split, mask in (("support", support), ("heldout", ~support)):
            pairs = [(i, j) for i in range(F) for j in range(F) if mask[i, j]]
            metrics = evaluate(method, loaded, xs, ys, pairs)
            rows.append({"condition": kind, "split": split, "world_or_seed": seed,
                         "method": method, "serialized_bytes": nbytes,
                         "parameter_tensor_bytes": sum(x.nbytes for x in state.values()),
                         "encode_wall_time_s": encode_wall, **metrics,
                         "payload_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                         "oracle_factor_ids": True})
    return rows
