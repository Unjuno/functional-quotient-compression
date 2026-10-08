"""Deterministic MA-258 synthetic expert-bank representation screen."""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np

E = 8
D = 16
N = 256
PSP_SEED = 2581601


def rotate_output(w: np.ndarray, angle: float) -> np.ndarray:
    """Apply a two-channel Givens view to output rows of an output-input matrix."""
    out = np.array(w, dtype=np.float32, copy=True)
    c, s = np.float32(np.cos(angle)), np.float32(np.sin(angle))
    a, b = out[0].copy(), out[1].copy()
    out[0] = c * a - s * b
    out[1] = s * a + c * b
    return out


def make_world(seed: int, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    angles = np.linspace(-0.7, 0.7, E, dtype=np.float32)
    if kind == "aligned":
        core = rng.normal(0, 1 / np.sqrt(D), (D, D)).astype(np.float32)
        teachers = np.stack([rotate_output(core, float(a)) for a in angles])
    elif kind == "unrelated":
        teachers = rng.normal(0, 1 / np.sqrt(D), (E, D, D)).astype(np.float32)
    else:
        raise ValueError(kind)
    x = rng.normal(size=(E, N, D)).astype(np.float32)
    y = np.einsum("eni,eoi->eno", x, teachers, optimize=True)
    return teachers, x, y, angles


def svd_codec(teachers: np.ndarray, rank: int) -> dict[str, np.ndarray]:
    bank = teachers.reshape(E, -1).astype(np.float64)
    base = bank.mean(axis=0)
    centered = bank - base
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:rank].astype(np.float32)
    coeff = (u[:, :rank] * s[:rank]).astype(np.float32)
    return {"base": base.astype(np.float32), "basis": basis, "coeff": coeff}


def psp_codec(teachers: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(PSP_SEED)
    contexts = rng.choice(np.array([-1, 1], dtype=np.int8), size=(E, D, D))
    superposed = np.sum(contexts.astype(np.float32) * teachers, axis=0)
    return {
        "superposed": superposed.astype(np.float32),
        "context_seed": np.array([PSP_SEED], dtype=np.uint64),
        "context_shape": np.array([E, D, D], dtype=np.uint16),
    }


def mirror_codec(teachers: np.ndarray, angles: np.ndarray) -> dict[str, np.ndarray]:
    # Post-fit oracle alignment is deliberately charged and disclosed.
    core = np.mean(np.stack([rotate_output(w, -float(a)) for w, a in zip(teachers, angles)]), axis=0)
    return {"core": core.astype(np.float32), "angles_f16": angles.astype(np.float16)}


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.ascontiguousarray(array), allow_pickle=False)
    return stream.getvalue()


def save_payload(path: Path, arrays: dict[str, np.ndarray]) -> int:
    """NPZ-compatible payload with fixed ZIP timestamps; returned size is authoritative."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            zf.writestr(info, _npy_bytes(arrays[name]))
    return path.stat().st_size


def load_payload(path: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(path) as zf:
        return {name[:-4]: np.load(io.BytesIO(zf.read(name)), allow_pickle=False)
                for name in sorted(zf.namelist())}


def decode(method: str, state: dict[str, np.ndarray], expert: int) -> np.ndarray:
    if method == "independent":
        return state["weights"][expert]
    if method == "tied":
        return state["weight"]
    if method == "svd":
        return (state["base"] + state["coeff"][expert] @ state["basis"]).reshape(D, D)
    if method == "psp":
        seed = int(state["context_seed"][0])
        shape = tuple(int(x) for x in state["context_shape"])
        contexts = np.random.default_rng(seed).choice(np.array([-1, 1], dtype=np.int8), size=shape)
        return contexts[expert].astype(np.float32) * state["superposed"]
    if method == "mirror":
        return rotate_output(state["core"], float(state["angles_f16"][expert]))
    raise ValueError(method)


def active_macs(method: str, rank: int = 0) -> int:
    # One expert, one input vector. Decoder arithmetic is included as a simple operation proxy.
    projection = D * D
    if method == "svd":
        return int(projection + 2 * rank * D * D)
    if method == "psp":
        return int(projection + D * D)  # elementwise unbinding
    if method == "mirror":
        return int(projection + 2 * D)  # two output rows rotated
    return projection


def evaluate(method: str, state: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray,
             teachers: np.ndarray, rank: int = 0, repeats: int = 3) -> dict[str, float | int]:
    started = time.perf_counter()
    raw = []
    normalized = []
    for rep in range(repeats):
        # First iteration scores; repeated iterations are solely timing calibration.
        for expert in range(E):
            w = decode(method, state, expert)
            pred = x[expert] @ w.T
            if rep == 0:
                err = float(np.mean((pred - y[expert]) ** 2))
                target = float(np.mean(y[expert] ** 2))
                raw.append(err)
                normalized.append(err / max(target, 1e-12))
    elapsed = time.perf_counter() - started
    return {
        "mse": float(np.mean(raw)),
        "normalized_mse": float(np.mean(normalized)),
        "wall_time_s": elapsed / repeats,
        "examples_per_s": float(E * N / max(elapsed / repeats, 1e-12)),
        "active_macs_per_example": active_macs(method, rank),
        "active_macs_total": active_macs(method, rank) * E * N,
        "examples": E * N,
        "optimizer_updates": 0,
    }


def make_states(teachers: np.ndarray, angles: np.ndarray, rank: int) -> dict[str, dict[str, np.ndarray]]:
    return {
        "independent": {"weights": teachers.astype(np.float32)},
        "tied": {"weight": teachers.mean(axis=0).astype(np.float32)},
        "svd": svd_codec(teachers, rank),
        "psp": psp_codec(teachers),
        "mirror": mirror_codec(teachers, angles),
    }


def make_state(method: str, teachers: np.ndarray, angles: np.ndarray, rank: int) -> dict[str, np.ndarray]:
    if method == "independent":
        return {"weights": teachers.astype(np.float32)}
    if method == "tied":
        return {"weight": teachers.mean(axis=0).astype(np.float32)}
    if method == "svd":
        return svd_codec(teachers, rank)
    if method == "psp":
        return psp_codec(teachers)
    if method == "mirror":
        return mirror_codec(teachers, angles)
    raise ValueError(method)


def run_world(seed: int, kind: str, rank: int, output_dir: Path) -> list[dict[str, object]]:
    teachers, x, y, angles = make_world(seed, kind)
    rows = []
    for method in ("independent", "tied", "svd", "psp", "mirror"):
        started = time.perf_counter()
        state = make_state(method, teachers, angles, rank)
        encode_seconds = time.perf_counter() - started
        path = output_dir / f"{kind}_{seed}_{method}.npz"
        byte_count = save_payload(path, state)
        reloaded = load_payload(path)
        metrics = evaluate(method, reloaded, x, y, teachers, rank=rank)
        rows.append({
            "condition": kind,
            "world_or_seed": seed,
            "method": method,
            "serialized_bytes": byte_count,
            "parameter_tensor_bytes": sum(a.nbytes for a in state.values()),
            "encode_wall_time_s": encode_seconds,
            **metrics,
            "status_note": f"rank={rank}; payload_sha256={__import__('hashlib').sha256(path.read_bytes()).hexdigest()}; oracle_expert_id",
        })
    return rows
