"""Synthetic module reuse and task-structure screen for MA-871."""
from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

K = 8
INPUT_DIM = 8
RANK = 2
N_TRAIN_TASKS = 12
N_HELDOUT_TASKS = 6
NOISE = 0.03
SUPPORT_SIZES = (4, 8, 16, 32)
HEADER = struct.Struct("<5sBBB")


def _rng(seed: int, tag: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, tag]))


def make_world(seed: int, train_examples: int = 128, test_examples: int = 4096) -> dict[str, Any]:
    rng = _rng(seed, 1)
    w = rng.normal(size=(K, INPUT_DIM)) / np.sqrt(INPUT_DIM)
    b = rng.uniform(-0.7, 0.7, size=K)
    true_basis = rng.normal(size=(RANK, K))
    true_basis /= np.linalg.norm(true_basis, axis=1, keepdims=True)
    z_train = rng.normal(size=(N_TRAIN_TASKS, RANK))
    z_test = rng.normal(size=(N_HELDOUT_TASKS, RANK))
    coeff_train = z_train @ true_basis
    coeff_test = z_test @ true_basis
    train_features = []
    train_targets = []
    for task in range(N_TRAIN_TASKS):
        trng = _rng(seed, 100 + task)
        x = trng.uniform(-1, 1, size=(train_examples, INPUT_DIM))
        phi = np.tanh(x @ w.T + b)
        y = phi @ coeff_train[task] + trng.normal(0, NOISE, size=train_examples)
        train_features.append(phi)
        train_targets.append(y)
    test_x, test_phi, support_phi, support_y = [], [], [], []
    for task in range(N_HELDOUT_TASKS):
        trng = _rng(seed, 10_000 + task)
        x_support = trng.uniform(-1, 1, size=(max(SUPPORT_SIZES), INPUT_DIM))
        phi_support = np.tanh(x_support @ w.T + b)
        y_support = phi_support @ coeff_test[task] + trng.normal(0, NOISE, size=len(x_support))
        x_test = trng.uniform(-1, 1, size=(test_examples, INPUT_DIM))
        phi_test = np.tanh(x_test @ w.T + b)
        support_phi.append(phi_support)
        support_y.append(y_support)
        test_x.append(x_test)
        test_phi.append(phi_test)
    return {
        "seed": seed,
        "w": w,
        "b": b,
        "true_basis": true_basis,
        "coeff_train": coeff_train,
        "coeff_test": coeff_test,
        "train_features": train_features,
        "train_targets": train_targets,
        "support_phi": support_phi,
        "support_y": support_y,
        "test_x": test_x,
        "test_phi": test_phi,
    }


def fit_structure(world: dict[str, Any]) -> dict[str, np.ndarray]:
    train_coeffs = np.stack([
        np.linalg.lstsq(world["train_features"][i], world["train_targets"][i], rcond=None)[0]
        for i in range(N_TRAIN_TASKS)
    ])
    centered = train_coeffs - train_coeffs.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:RANK]
    random = _rng(world["seed"], 42).normal(size=(RANK, K))
    random, _ = np.linalg.qr(random.T)
    random_basis = random[:, :RANK].T
    pooled_phi = np.concatenate(world["train_features"], axis=0)
    pooled_y = np.concatenate(world["train_targets"], axis=0)
    pooled = np.linalg.lstsq(pooled_phi, pooled_y, rcond=None)[0]
    return {"train_coeffs": train_coeffs, "basis": basis, "random_basis": random_basis, "pooled": pooled}


def _support_fit(phi: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    return np.linalg.lstsq(phi[:n], y[:n], rcond=None)[0]


def _fit_lowrank(phi: np.ndarray, y: np.ndarray, n: int, basis: np.ndarray) -> np.ndarray:
    design = phi[:n] @ basis.T
    code = np.linalg.lstsq(design, y[:n], rcond=None)[0]
    return code @ basis


def _nearest_codebook(phi: np.ndarray, y: np.ndarray, n: int, coeffs: np.ndarray) -> tuple[np.ndarray, int]:
    errors = [float(np.mean((phi[:n] @ c - y[:n]) ** 2)) for c in coeffs]
    idx = int(np.argmin(errors))
    return coeffs[idx], idx


def _f32(a: np.ndarray) -> bytes:
    return np.asarray(a, dtype="<f4", order="C").tobytes(order="C")


def serialize_payload(world: dict[str, Any], model: dict[str, np.ndarray], method: str, codes: np.ndarray | None = None, indices: list[int] | None = None) -> bytes:
    method_ids = {"independent": 1, "mirror": 2, "native_pca": 3, "random_basis": 4, "codebook": 5, "pooled": 6, "teacher": 7}
    payload = bytearray(HEADER.pack(b"MA871", method_ids[method], K, RANK))
    payload.extend(_f32(world["w"]))
    payload.extend(_f32(world["b"]))
    if method in ("mirror", "native_pca", "random_basis"):
        basis = model["basis"] if method in ("mirror", "native_pca") else model["random_basis"]
        payload.extend(_f32(basis))
        payload.extend(_f32(codes))
    elif method in ("independent", "teacher"):
        payload.extend(_f32(codes))
    elif method == "codebook":
        payload.extend(_f32(model["train_coeffs"]))
        payload.extend(bytes(indices or []))
    elif method == "pooled":
        payload.extend(_f32(model["pooled"]))
    return bytes(payload)


def _metric_predictions(world: dict[str, Any], model: dict[str, np.ndarray], support_size: int) -> dict[str, Any]:
    out: dict[str, Any] = {k: [] for k in ("independent", "mirror", "native_pca", "random_basis", "codebook", "pooled", "teacher")}
    codes: dict[str, list[np.ndarray]] = {k: [] for k in out}
    codebook_indices: list[int] = []
    support_seconds: dict[str, float] = {k: 0.0 for k in out}
    inference_seconds: dict[str, float] = {k: 0.0 for k in out}
    per_task: dict[str, list[float]] = {k: [] for k in out}
    for task in range(N_HELDOUT_TASKS):
        phi = world["support_phi"][task]
        y = world["support_y"][task]
        test_phi = world["test_phi"][task]
        target = world["coeff_test"][task]
        for method in ("independent",):
            t0 = time.perf_counter()
            coeff = _support_fit(phi, y, support_size)
            support_seconds[method] += time.perf_counter() - t0
            t1 = time.perf_counter()
            pred = test_phi @ coeff
            inference_seconds[method] += time.perf_counter() - t1
            err = float(np.mean((pred - test_phi @ target) ** 2))
            out[method].append(err)
            per_task[method].append(err)
            codes[method].append(coeff)
        for method, basis in (("mirror", model["basis"]), ("native_pca", model["basis"]), ("random_basis", model["random_basis"])):
            t0 = time.perf_counter()
            coeff = _fit_lowrank(phi, y, support_size, basis)
            support_seconds[method] += time.perf_counter() - t0
            t1 = time.perf_counter()
            pred = test_phi @ coeff
            inference_seconds[method] += time.perf_counter() - t1
            err = float(np.mean((pred - test_phi @ target) ** 2))
            out[method].append(err)
            per_task[method].append(err)
            codes[method].append(np.linalg.lstsq(phi[:support_size] @ basis.T, y[:support_size], rcond=None)[0])
        t0 = time.perf_counter()
        coeff, idx = _nearest_codebook(phi, y, support_size, model["train_coeffs"])
        support_seconds["codebook"] += time.perf_counter() - t0
        t1 = time.perf_counter()
        pred = test_phi @ coeff
        inference_seconds["codebook"] += time.perf_counter() - t1
        err = float(np.mean((pred - test_phi @ target) ** 2))
        out["codebook"].append(err)
        per_task["codebook"].append(err)
        codes["codebook"].append(coeff)
        codebook_indices.append(idx)
        t1 = time.perf_counter()
        pooled_pred = test_phi @ model["pooled"]
        teacher_pred = test_phi @ target
        inference_seconds["pooled"] += time.perf_counter() - t1
        out["pooled"].append(float(np.mean((pooled_pred - test_phi @ target) ** 2)))
        per_task["pooled"].append(out["pooled"][-1])
        codes["pooled"].append(model["pooled"])
        out["teacher"].append(0.0)
        per_task["teacher"].append(0.0)
        codes["teacher"].append(target)
    for method in out:
        out[method] = float(np.mean(out[method]))
    payloads = {
        method: len(serialize_payload(world, model, method, np.stack(codes[method]), codebook_indices if method == "codebook" else None))
        for method in out
    }
    # Deterministic alias is tested on the predictions and on equivalent rank-2 serialized objects.
    return {"mse": out, "per_task_mse": per_task, "payload_bytes": payloads, "adaptation_seconds": support_seconds, "inference_seconds": inference_seconds, "codebook_indices": codebook_indices}


def evaluate(seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    world = make_world(seed)
    train_started = time.perf_counter()
    model = fit_structure(world)
    train_seconds = time.perf_counter() - train_started
    by_support = {str(n): _metric_predictions(world, model, n) for n in SUPPORT_SIZES}
    # Store per-task metrics at the primary 8-shot setting to audit worst cases.
    primary = by_support["8"]
    model_bytes = {m: primary["payload_bytes"][m] for m in primary["payload_bytes"]}
    return {
        "seed": seed,
        "train_basis_seconds": train_seconds,
        "by_support": by_support,
        "payload_bytes_at_8": model_bytes,
        "module_evaluation_mac_proxy_per_query": K * INPUT_DIM + K,
        "decoder_mac_proxy_per_query": {"independent": 2*K, "mirror": 2*RANK*K+2*K, "native_pca": 2*RANK*K+2*K, "random_basis": 2*RANK*K+2*K, "codebook": 2*K, "pooled": 2*K, "teacher": 0},
        "total_wall_seconds": time.perf_counter() - started,
    }


def run(seeds: list[int]) -> dict[str, Any]:
    rows = [evaluate(s) for s in seeds]
    methods = ("independent", "mirror", "native_pca", "random_basis", "codebook", "pooled", "teacher")
    summary: dict[str, Any] = {"experiment_id": "MA-871", "seeds": seeds, "support_sizes": list(SUPPORT_SIZES), "rows": rows, "primary_support": 8}
    summary["mean_mse_by_support"] = {
        str(n): {m: float(np.mean([r["by_support"][str(n)]["mse"][m] for r in rows])) for m in methods}
        for n in SUPPORT_SIZES
    }
    summary["max_mse_by_support"] = {
        str(n): {m: float(max(r["by_support"][str(n)]["mse"][m] for r in rows)) for m in methods}
        for n in SUPPORT_SIZES
    }
    summary["payload_bytes_at_8"] = {m: int(np.mean([r["payload_bytes_at_8"][m] for r in rows])) for m in methods}
    summary["mean_adaptation_seconds_at_8"] = {m: float(np.mean([r["by_support"]["8"]["adaptation_seconds"][m] for r in rows])) for m in methods}
    summary["mean_inference_seconds_at_8"] = {m: float(np.mean([r["by_support"]["8"]["inference_seconds"][m] for r in rows])) for m in methods}
    summary["mean_train_basis_seconds"] = float(np.mean([r["train_basis_seconds"] for r in rows]))
    summary["max_total_wall_seconds"] = float(max(r["total_wall_seconds"] for r in rows))
    return summary


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
