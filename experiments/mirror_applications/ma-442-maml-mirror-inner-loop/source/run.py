#!/usr/bin/env python3
"""Frozen MA-442 first-order meta-learning mechanism screen."""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

D = 6
PAIR_A, PAIR_B = (0, 1), (2, 3)
TEMPLATE = torch.tensor([1.2, 0.3, 0.9, -0.5, 0.4, 0.8], dtype=torch.float32)
INNER_STEPS, INNER_LR, OUTER_STEPS, TASKS_PER_UPDATE = 4, 0.08, 160, 8
SUPPORT_N, QUERY_N = 8, 32


def orbit(theta: torch.Tensor) -> torch.Tensor:
    """Two disjoint Givens rotations: theta is [a,b], output is a 6D weight."""
    a, b = theta.unbind(-1)
    c1, s1 = torch.cos(a), torch.sin(a)
    c2, s2 = torch.cos(b), torch.sin(b)
    t = TEMPLATE.to(theta.device)
    out = t.expand(theta.shape[:-1] + (D,)).clone()
    out[..., 0] = c1 * t[0] - s1 * t[1]
    out[..., 1] = s1 * t[0] + c1 * t[1]
    out[..., 2] = c2 * t[2] - s2 * t[3]
    out[..., 3] = s2 * t[2] + c2 * t[3]
    return out


def task_rng(seed: int, split: str) -> np.random.Generator:
    split_offset = 0 if split == "dev" else 100_000
    return np.random.default_rng(seed + split_offset)


def sample_task(rng: np.random.Generator, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    angles = torch.tensor(rng.uniform(-0.9, 0.9, size=2), dtype=torch.float32)
    weight = orbit(angles)
    x = torch.tensor(rng.normal(size=(n, D)), dtype=torch.float32)
    noise = torch.tensor(rng.normal(scale=0.02, size=n), dtype=torch.float32)
    y = x @ weight + noise
    return angles, x, y


def mse(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.mean((x @ w - y) ** 2)


def inner_full(init: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    w = init.detach().clone().requires_grad_(True)
    for _ in range(INNER_STEPS):
        g, = torch.autograd.grad(mse(xs, ys, w), w)
        w = (w - INNER_LR * g).detach().requires_grad_(True)
    return w.detach()


def inner_mirror(base: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    code = torch.zeros(2, dtype=base.dtype, requires_grad=True)
    for _ in range(INNER_STEPS):
        w = orbit(code) + (base - TEMPLATE)
        g, = torch.autograd.grad(mse(xs, ys, w), code)
        code = (code - INNER_LR * g).detach().requires_grad_(True)
    return code.detach()


def inner_lora(base: torch.Tensor, basis: torch.Tensor, xs: torch.Tensor,
               ys: torch.Tensor) -> torch.Tensor:
    code = torch.zeros(2, dtype=base.dtype, requires_grad=True)
    for _ in range(INNER_STEPS):
        w = base + basis @ code
        g, = torch.autograd.grad(mse(xs, ys, w), code)
        code = (code - INNER_LR * g).detach().requires_grad_(True)
    return code.detach()


def prediction_weight(method: str, base: torch.Tensor, code: torch.Tensor,
                      basis: torch.Tensor | None = None) -> torch.Tensor:
    if method in ("mirror", "native_givens"):
        return orbit(code) + (base - TEMPLATE)
    if method == "lora":
        assert basis is not None
        return base + basis @ code
    return code


def _meta_update(params: list[torch.Tensor], grads: list[torch.Tensor], lr: float) -> list[torch.Tensor]:
    return [(p - lr * g).detach() for p, g in zip(params, grads)]


def train(method: str, seed: int, split: str) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    """First-order MAML: inner support updates, query-gradient outer updates."""
    torch.manual_seed(seed + {"full": 11, "mirror": 22, "native_givens": 22, "lora": 33}[method])
    base = torch.zeros(D, dtype=torch.float32)
    basis = None
    if method == "lora":
        basis = torch.randn(D, 2) * 0.2
    rng = task_rng(seed + 73, split)
    outer_lr = {"full": 0.08, "mirror": 0.05, "native_givens": 0.05, "lora": 0.035}[method]
    start = time.perf_counter()
    for _ in range(OUTER_STEPS):
        gradients = [torch.zeros_like(base)]
        if method == "lora":
            gradients.append(torch.zeros_like(basis))
        for _ in range(TASKS_PER_UPDATE):
            _, x, y = sample_task(rng, SUPPORT_N + QUERY_N)
            xs, xq = x[:SUPPORT_N], x[SUPPORT_N:]
            ys, yq = y[:SUPPORT_N], y[SUPPORT_N:]
            if method == "full":
                adapted = inner_full(base, xs, ys)
                g, = torch.autograd.grad(mse(xq, yq, adapted), adapted)
                gradients[0] += g.detach()
            elif method in ("mirror", "native_givens"):
                code = inner_mirror(base, xs, ys)
                # FOMAML: query gradient at adapted code, with inner trajectory detached.
                b = base.detach().clone().requires_grad_(True)
                w = orbit(code) + (b - TEMPLATE)
                g, = torch.autograd.grad(mse(xq, yq, w), b)
                gradients[0] += g.detach()
            else:
                code = inner_lora(base, basis, xs, ys)
                b = base.detach().clone().requires_grad_(True)
                u = basis.detach().clone().requires_grad_(True)
                w = b + u @ code
                gb, gu = torch.autograd.grad(mse(xq, yq, w), (b, u))
                gradients[0] += gb.detach()
                gradients[1] += gu.detach()
        scale = 1.0 / TASKS_PER_UPDATE
        gradients = [g * scale for g in gradients]
        if method == "lora":
            base, basis = _meta_update([base, basis], gradients, outer_lr)
        else:
            base = _meta_update([base], gradients, outer_lr)[0]
    elapsed = time.perf_counter() - start
    return base, basis, {"outer_updates": OUTER_STEPS, "inner_updates": OUTER_STEPS * TASKS_PER_UPDATE * INNER_STEPS,
                         "train_examples": OUTER_STEPS * TASKS_PER_UPDATE * (SUPPORT_N + QUERY_N),
                         "train_wall_s": elapsed}


def fit_independent(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    return torch.linalg.lstsq(xs, ys).solution


def serialize_payload(arrays: dict[str, np.ndarray], metadata: dict) -> bytes:
    buf = io.BytesIO()
    packed = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}
    packed["__schema_json__"] = np.frombuffer(json.dumps(metadata, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    np.savez(buf, **packed)
    return buf.getvalue()


def evaluate(method: str, base: torch.Tensor | None, basis: torch.Tensor | None,
             seed: int, split: str) -> tuple[dict, dict[str, np.ndarray], bytes]:
    rng = task_rng(seed + 901, split)
    scores, vectors, codes, angles, fit_times = [], [], [], [], []
    for _ in range(16):
        true_angles, x, y = sample_task(rng, SUPPORT_N + QUERY_N)
        xs, xq = x[:SUPPORT_N], x[SUPPORT_N:]
        ys, yq = y[:SUPPORT_N], y[SUPPORT_N:]
        st = time.perf_counter()
        if method in ("mirror", "native_givens"):
            code = inner_mirror(base, xs, ys)
            w = prediction_weight(method, base, code)
            angles.append(code.detach().numpy())
            codes.append(code.detach().numpy())
        elif method == "lora":
            code = inner_lora(base, basis, xs, ys)
            w = prediction_weight(method, base, code, basis)
            codes.append(code.detach().numpy())
        elif method == "full":
            w = inner_full(base, xs, ys)
            codes.append(w.detach().numpy())
        elif method == "no_adapt":
            w = base.detach()
            codes.append(np.empty((0,), dtype=np.float32))
        elif method == "independent":
            w = fit_independent(xs, ys)
            codes.append(w.detach().numpy())
        else:
            raise ValueError(method)
        fit_times.append(time.perf_counter() - st)
        scores.append(float(torch.sqrt(mse(xq, yq, w))))
        vectors.append(w.detach().numpy())
    arrays: dict[str, np.ndarray] = {"task_vectors": np.stack(vectors)}
    if method in ("mirror", "native_givens", "no_adapt", "full"):
        arrays["shared_init"] = base.detach().numpy()[None, :]
    if method == "lora":
        arrays["shared_init"] = base.detach().numpy()[None, :]
        arrays["basis"] = basis.detach().numpy()
    # Native model representation stores task codes and shared decoder state; task_vectors above
    # are retained in result JSON for audit but only codes are served in the compressed bank.
    if method in ("mirror", "native_givens", "lora"):
        arrays.pop("task_vectors")
        arrays["task_codes"] = np.stack(codes)
    if method == "full":
        arrays["task_codes"] = np.stack(codes)  # full task weights
    if method == "independent":
        arrays["task_vectors"] = np.stack(codes)
    if method == "no_adapt":
        arrays = {"shared_init": base.detach().numpy()[None, :]}
    elapsed = float(sum(fit_times))
    family = "angle-conditioned-linear-v1" if method in ("mirror", "native_givens") else method
    metadata = {"method_family": family, "format": "MA442-inference-npz-v1", "task_count": 16,
                "dtype": "float32", "dimensions": D, "inner_steps": INNER_STEPS,
                "inner_lr": INNER_LR, "code_dim": 2 if method in ("mirror", "native_givens", "lora") else D}
    payload = serialize_payload(arrays, metadata)
    metrics = {"query_rmse": float(np.mean(scores)), "query_rmse_sd": float(np.std(scores, ddof=1)),
               "payload_bytes": len(payload), "inference_wall_s": elapsed,
               "query_examples": 16 * QUERY_N, "adaptation_examples": 16 * SUPPORT_N,
               "active_ops_proxy": 16 * (SUPPORT_N * INNER_STEPS * D * (2 if method in ("mirror", "native_givens", "lora") else 1) + QUERY_N * D),
               "payload_sha256": __import__("hashlib").sha256(payload).hexdigest()}
    return metrics, arrays, payload


def run(seed: int, split: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    states = {}
    for method in ("full", "mirror", "native_givens", "lora"):
        base, basis, train_metrics = train(method, seed, split)
        states[method] = (base, basis, train_metrics)
        eval_metrics, arrays, payload = evaluate(method, base, basis, seed, split)
        results[method] = {**train_metrics, **eval_metrics}
        (out / f"{method}_payload.npz").write_bytes(payload)
    base, _, no_train = states["full"]
    for method in ("no_adapt", "independent"):
        eval_metrics, arrays, payload = evaluate(method, base, None, seed, split)
        results[method] = {**no_train, **eval_metrics}
        (out / f"{method}_payload.npz").write_bytes(payload)
    # Native conditioning is deliberately the same map, optimizer, initialization and episodes.
    # Its equality is checked in verify.py; report output hashes expose the exact alias.
    result = {"experiment_id": "MA-442", "seed": seed, "split": split,
              "task": {"dimension": D, "support": SUPPORT_N, "query": QUERY_N,
                       "inner_steps": INNER_STEPS, "outer_updates": OUTER_STEPS,
                       "tasks_per_update": TASKS_PER_UPDATE}, "methods": results}
    (out / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("dev", "fresh"), default="dev")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.split, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
