"""MA-255 amendment A1: correlated task models make PSP utility testable."""
import csv
import hashlib
import io
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
PAY = ART / "payloads_a1"
D, N, RANK, Q = 64, 16, 8, 512
DEV, FRESH, SEEDS = (25520, 25521), (25530, 25531, 25532), (0, 1, 2)
STEPS = 1600
LR_GRID = (0.001, 0.01, 0.03)


def data(world, seed):
    g = torch.Generator().manual_seed(world * 100003 + seed * 7919 + 255)
    basis = torch.randn(RANK, D, generator=g) / D**0.5
    code = torch.randn(N, RANK, generator=g)
    targets = code @ basis + 0.03 * torch.randn(N, D, generator=g)
    x = torch.randn(Q, D, generator=g)
    return targets, x


def nrmse(pred, target, x):
    pred_out = x @ pred.T
    target_out = x @ target.T
    return float(torch.mean((pred_out - target_out) ** 2).sqrt() /
                 torch.mean(target_out ** 2).sqrt())


def serialize(state):
    b = io.BytesIO()
    torch.save(state, b)
    return b.getvalue()


def fit(targets, rank, lr, seed):
    torch.manual_seed(seed)
    basis = torch.nn.Parameter(torch.randn(rank, D) * 0.01)
    code = torch.nn.Parameter(torch.randn(N, rank) * 0.01)
    opt = torch.optim.Adam([basis, code], lr=lr)
    for _ in range(STEPS):
        opt.zero_grad()
        loss = ((code @ basis - targets) ** 2).mean()
        loss.backward()
        opt.step()
    return code.detach(), basis.detach()


def payload(method, world, seed, task, pred, targets, code=None, basis=None):
    if method == "independent":
        state = {"task_vector": pred[task].half(), "task": task,
                 "metadata": {"method": method, "world": world, "seed": seed}}
    elif method == "shared":
        state = {"shared_mean": targets.mean(0).half(), "task": task,
                 "metadata": {"method": method, "world": world, "seed": seed}}
    else:
        state = {"shared_basis": basis.half(), "task_code": code[task].half(),
                 "task": task, "metadata": {"method": method, "rank": int(basis.shape[0]),
                 "world": world, "seed": seed}}
    return serialize(state)


def run(phase):
    PAY.mkdir(parents=True, exist_ok=True)
    rows = []
    worlds = DEV if phase == "development" else FRESH
    for world in worlds:
        for seed in SEEDS:
            targets, x = data(world, seed)
            # Exact native PSP unbinding: W = sum_t c_t outer v_t;
            # estimate v_i as D * W^T c_i / ||c_i||^2.
            g = torch.Generator().manual_seed(world + seed + 919)
            signs = torch.randint(0, 2, (N, D), generator=g).float() * 2 - 1
            psp_codes = signs / D**0.5
            superposed = psp_codes.T @ targets
            psp_pred = (psp_codes @ superposed) / psp_codes.square().sum(1, keepdim=True)
            rows += record("psp", world, seed, targets, x, psp_pred, psp_codes, None, None, 0.0, 0)

            mean = targets.mean(0, keepdim=True).expand(N, -1)
            rows += record("shared", world, seed, targets, x, mean, None, None, None, 0.0, 0)
            rows += record("independent", world, seed, targets, x, targets, None, None, None, 0.0, 0)

            for rank in (RANK,):
                for lr in (LR_GRID if phase == "development" else (0.01,)):
                    t0 = time.perf_counter()
                    code, basis = fit(targets, rank, lr, world * 17 + seed)
                    elapsed = time.perf_counter() - t0
                    pred = code @ basis
                    rows += record(f"mirror_r{rank}_lr{lr:g}", world, seed, targets, x,
                                   pred, None, code, basis, elapsed, STEPS)
                    # Generic learned low-rank code is the functionally identical
                    # simple control; retain separately named payload/metric rows.
                    rows += record(f"generic_lowrank_r{rank}_lr{lr:g}", world, seed,
                                   targets, x, pred, None, code, basis, elapsed, STEPS)
    out = ROOT / f"{phase.upper()}_RESULTS_A1.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"phase": phase, "rows": len(rows), "path": str(out)}))


def record(method, world, seed, targets, x, pred, psp_codes, code, basis, elapsed, updates):
    result = []
    for task in range(N):
        if method == "psp":
            state = {"superposed": (psp_codes.T @ targets).half(),
                     "codes": psp_codes.half(), "task": task,
                     "metadata": {"method": method, "world": world, "seed": seed,
                                  "decoder": "W @ c / ||c||^2"}}
        else:
            state = payload(method, world, seed, task, pred, targets, code, basis)
        blob = serialize(state)
        path = PAY / f"{world}_{seed}_{method}_{task}.pt"
        path.write_bytes(blob)
        target = targets[task:task + 1]
        result.append({"world": world, "seed": seed, "method": method, "task": task,
                       "nrmse": nrmse(pred[task:task + 1], target, x),
                       "payload_bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
                       "path": str(path.relative_to(ROOT.parents[2])), "train_seconds": elapsed,
                       "updates": updates, "query_examples": Q,
                       "active_compute_proxy": D * Q if method in ("psp", "shared", "independent") else
                       2 * int(basis.shape[0]) * D * Q})
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "fresh"), required=True)
    run(parser.parse_args().phase)
