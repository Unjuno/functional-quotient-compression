"""MA-381 aligned LoRAHub screen, with signed few-shot coefficient search."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import (CANDIDATES, INPUT_DIM, OUTPUT_DIM, RANK, TARGETS,
                   LoRABank, compose, mac_proxy, rotation_matrices)

torch.set_num_threads(1)
SOURCE_UPDATES = 1200
COMPOSITION_UPDATES = 300
BATCH = 64
LR = 0.01
SUPPORT = 64


def make_world(seed: int) -> dict[str, object]:
    g = torch.Generator().manual_seed(seed)
    a0 = torch.randn(RANK, INPUT_DIM, generator=g) / INPUT_DIM**0.5
    b0 = 0.4 * torch.randn(OUTPUT_DIM, RANK, generator=g) / RANK**0.5
    angles = (2 * torch.rand(CANDIDATES, 1, generator=g) - 1) * np.pi
    q = rotation_matrices(angles)
    matrices = b0[None] @ q @ a0[None]
    target_coeff = torch.zeros(TARGETS, CANDIDATES)
    for t in range(TARGETS):
        ids = torch.randperm(CANDIDATES, generator=g)[:3]
        vals = (2 * torch.rand(3, generator=g) - 1)
        target_coeff[t, ids] = vals
    data: dict[str, object] = {"matrices": matrices, "target_coeff": target_coeff}
    x_source = torch.randn(4096, INPUT_DIM, generator=g)
    data["x_source_train"] = x_source
    data["source_train"] = torch.einsum("bd,ndk->bnk", x_source, matrices.transpose(1, 2))
    for split, count in (("validation", 1024), ("test", 2048)):
        x = torch.randn(count, INPUT_DIM, generator=g)
        data[f"x_source_{split}"] = x
        data[f"source_{split}"] = torch.einsum("bd,ndk->bnk", x, matrices.transpose(1, 2))
    for split, count in (("support", SUPPORT), ("test", 2048)):
        x = torch.randn(TARGETS, count, INPUT_DIM, generator=g)
        flat = x.reshape(-1, INPUT_DIM)
        src = torch.einsum("bd,ndk->bnk", flat, matrices.transpose(1, 2))
        src = src.reshape(TARGETS, count, CANDIDATES, OUTPUT_DIM)
        y = torch.einsum("tn,tbnd->tbd", target_coeff, src)
        data[f"x_target_{split}"] = x
        data[f"target_{split}"] = y
    return data


def nrmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2) / torch.mean(target**2)))


def fit_source(method: str, seed: int, world: dict[str, object]):
    bank = LoRABank(method, seed + 1000)
    opt = torch.optim.Adam(bank.parameters(), lr=LR)
    x, target = world["x_source_train"], world["source_train"]
    start = time.perf_counter()
    for _ in range(SOURCE_UPDATES):
        ids = torch.randint(len(x), (BATCH,), generator=torch.Generator().manual_seed(seed + 2000 + _))
        pred = bank(x[ids])
        loss = torch.mean((pred - target[ids]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    elapsed = time.perf_counter() - start
    metrics = {}
    bank.eval()
    with torch.no_grad():
        for split in ("train", "validation", "test"):
            metrics[f"source_{split}_nrmse"] = nrmse(
                bank(world[f"x_source_{split}"]), world[f"source_{split}"])
    return bank, metrics, elapsed


def fit_composition(bank: LoRABank, seed: int, world: dict[str, object]):
    x_support = world["x_target_support"]
    x_flat = x_support.reshape(-1, INPUT_DIM)
    with torch.no_grad():
        source_support = bank(x_flat).reshape(TARGETS, SUPPORT, CANDIDATES, OUTPUT_DIM).detach()
        source_test = bank(world["x_target_test"].reshape(-1, INPUT_DIM)).reshape(
            TARGETS, 2048, CANDIDATES, OUTPUT_DIM).detach()
    coeff = torch.nn.Parameter(torch.zeros(TARGETS, CANDIDATES))
    opt = torch.optim.Adam([coeff], lr=LR)
    start = time.perf_counter()
    for _ in range(COMPOSITION_UPDATES):
        pred = torch.einsum("tn,tbnd->tbd", coeff, source_support)
        loss = torch.mean((pred - world["target_support"]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    elapsed = time.perf_counter() - start
    with torch.no_grad():
        support_pred = torch.einsum("tn,tbnd->tbd", coeff, source_support)
        test_pred = torch.einsum("tn,tbnd->tbd", coeff, source_test)
        support_by_target = [nrmse(support_pred[t], world["target_support"][t]) for t in range(TARGETS)]
        test_by_target = [nrmse(test_pred[t], world["target_test"][t]) for t in range(TARGETS)]
    metrics = {
        "target_support_nrmse_by_task": support_by_target,
        "target_support_nrmse": float(np.mean(support_by_target)),
        "target_test_nrmse_by_task": test_by_target,
        "target_test_nrmse": float(np.mean(test_by_target)),
    }
    return coeff.detach(), metrics, elapsed


def gauge_audit(bank: LoRABank, seed: int) -> dict[str, float] | None:
    """Apply non-orthogonal GL(2) reparameterizations and compare functions."""
    if bank.method != "independent":
        return None
    ggen = torch.Generator().manual_seed(seed + 9000)
    x = torch.randn(128, INPUT_DIM, generator=ggen, dtype=torch.float64)
    coeff = torch.randn(TARGETS, CANDIDATES, generator=ggen, dtype=torch.float64)
    with torch.no_grad():
        a = bank.down.detach().double()
        b = bank.up.detach().double()
        matrices = b @ a
        transformed_a, transformed_b = [], []
        for i in range(CANDIDATES):
            # Nonsingular, deliberately non-orthogonal matrix with task jitter.
            g = torch.tensor([[1.8, 0.25], [-0.12, 0.72]], dtype=torch.float64)
            g = g + 0.025 * torch.randn(2, 2, generator=ggen, dtype=torch.float64)
            if torch.abs(torch.det(g)) < 0.5:
                g[0, 0] += 0.5
            transformed_b.append(b[i] @ g)
            transformed_a.append(torch.linalg.solve(g, a[i]))
        a2, b2 = torch.stack(transformed_a), torch.stack(transformed_b)
        matrices2 = b2 @ a2
        max_matrix = float(torch.max(torch.abs(matrices - matrices2)))
        y1 = torch.einsum("bd,ndk->bnk", x, matrices.transpose(1, 2))
        y2 = torch.einsum("bd,ndk->bnk", x, matrices2.transpose(1, 2))
        max_output = float(torch.max(torch.abs(y1 - y2)))
        c1 = torch.einsum("tn,bnd->tbd", coeff, y1)
        c2 = torch.einsum("tn,bnd->tbd", coeff, y2)
        max_comp = float(torch.max(torch.abs(c1 - c2)))
    return {"max_delta_matrix": max_matrix, "max_delta_output": max_output, "max_delta_composed": max_comp}


def save_payload(path: Path, bank: LoRABank, coeff: torch.Tensor) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bank.payload_arrays(coeff))
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def run(seed: int, condition: str, outdir: Path) -> dict[str, object]:
    world = make_world(seed)
    rows = []
    for method in LoRABank.METHODS:
        start = time.perf_counter()
        bank, src_metrics, src_s = fit_source(method, seed, world)
        coeff, comp_metrics, comp_s = fit_composition(bank, seed, world)
        gauge = gauge_audit(bank, seed)
        path = outdir / f"{condition}_{seed}_{method}.npz"
        size, digest = save_payload(path, bank, coeff)
        with torch.no_grad():
            x = world["x_target_test"].reshape(-1, INPUT_DIM)
            t0 = time.perf_counter()
            source = bank(x).reshape(TARGETS, 2048, CANDIDATES, OUTPUT_DIM)
            _ = torch.einsum("tn,tbnd->tbd", coeff, source)
            infer_s = time.perf_counter() - t0
        rows.append({
            "method": method,
            "payload_path": path.name,
            "serialized_bytes": size,
            "payload_sha256": digest,
            **src_metrics,
            **comp_metrics,
            "source_mac_proxy": mac_proxy(method),
            "composition_mac_proxy_per_example": CANDIDATES * OUTPUT_DIM,
            "composition_search_mac_proxy_per_update": SUPPORT * CANDIDATES * OUTPUT_DIM,
            "optimizer_updates": SOURCE_UPDATES + COMPOSITION_UPDATES,
            "source_examples_seen": SOURCE_UPDATES * BATCH * CANDIDATES,
            "target_support_examples_seen": COMPOSITION_UPDATES * SUPPORT * TARGETS,
            "source_training_wall_s": src_s,
            "composition_wall_s": comp_s,
            "total_training_wall_s": time.perf_counter() - start,
            "inference_examples_per_s": (TARGETS * 2048) / max(infer_s, 1e-9),
            "gauge_audit": gauge,
        })
    return {"experiment_id": "MA-381", "condition": condition, "seed": seed,
            "source_updates": SOURCE_UPDATES, "composition_updates_per_target": COMPOSITION_UPDATES,
            "support_examples_per_target": SUPPORT, "learning_rate": LR, "batch_size": BATCH,
            "results": rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--condition", choices=("development", "fresh"), required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--json", type=Path, required=True)
    args = p.parse_args()
    result = run(args.seed, args.condition, args.outdir)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"seed": args.seed, "condition": args.condition,
                      "results": [{k: r[k] for k in ("method", "serialized_bytes", "source_test_nrmse", "target_test_nrmse", "total_training_wall_s")}
                                  for r in result["results"]]}, indent=2))


if __name__ == "__main__":
    main()
