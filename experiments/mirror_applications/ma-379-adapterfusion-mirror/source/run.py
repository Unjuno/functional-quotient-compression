"""CPU screening runner for MA-379; protocol is frozen in the parent commit."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import INPUT_DIM, OUTPUT_DIM, RANK, SOURCES, TARGETS, PLANES, SourceBank, rotation_matrices

torch.set_num_threads(1)

SOURCE_UPDATES = 1200
FUSION_UPDATES = 400
BATCH = 64
LR = 0.01


def make_world(seed: int) -> dict[str, object]:
    g = torch.Generator().manual_seed(seed)
    a0 = torch.randn(RANK, INPUT_DIM, generator=g) / INPUT_DIM**0.5
    b0 = 0.4 * torch.randn(OUTPUT_DIM, RANK, generator=g) / RANK**0.5
    true_angles = (2 * torch.rand(SOURCES, len(PLANES), generator=g) - 1) * np.pi
    q = rotation_matrices(true_angles)
    matrices = b0[None, :, :] @ q @ a0[None, :, :]
    teacher_router = 1.5 * torch.randn(TARGETS, INPUT_DIM, SOURCES, generator=g) / INPUT_DIM**0.5
    data: dict[str, object] = {"matrices": matrices, "teacher_router": teacher_router}
    for split, count in (("train", 4096), ("validation", 1024), ("test", 2048)):
        x = torch.randn(count, INPUT_DIM, generator=g)
        src = torch.einsum("bd,ndk->bnk", x, matrices.transpose(1, 2))
        weights = torch.softmax(torch.einsum("bd,tdn->btn", x, teacher_router), dim=-1)
        target = torch.einsum("btn,bnd->btd", weights, src)
        data[f"x_{split}"] = x
        data[f"source_{split}"] = src
        data[f"target_{split}"] = target
    return data


def nrmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((actual - expected) ** 2) / torch.mean(expected**2)))


def fit_bank(method: str, seed: int, world: dict[str, object]) -> tuple[SourceBank, dict[str, float], float]:
    model = SourceBank(method, seed + 1000).cpu()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    x_train = world["x_train"]
    source_train = world["source_train"]
    # Identical sampled examples for every method in a world.
    g = torch.Generator().manual_seed(seed + 2000)
    start = time.perf_counter()
    for _ in range(SOURCE_UPDATES):
        ids = torch.randint(len(x_train), (BATCH,), generator=g)
        pred = model(x_train[ids])
        loss = torch.mean((pred - source_train[ids]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    metrics = {}
    with torch.no_grad():
        for split in ("train", "validation", "test"):
            metrics[f"source_{split}_nrmse"] = nrmse(model(world[f"x_{split}"]), world[f"source_{split}"])
    elapsed = time.perf_counter() - start
    return model.eval(), metrics, elapsed


def fit_fusion(model: SourceBank, seed: int, world: dict[str, object]) -> tuple[torch.Tensor, dict[str, object], float]:
    x_train = world["x_train"]
    with torch.no_grad():
        src_train = model(x_train).detach()
        src_validation = model(world["x_validation"]).detach()
        src_test = model(world["x_test"]).detach()
    router = torch.nn.Parameter(torch.zeros(TARGETS, INPUT_DIM, SOURCES))
    opt = torch.optim.Adam([router], lr=LR)
    g = torch.Generator().manual_seed(seed + 3000)
    start = time.perf_counter()
    for _ in range(FUSION_UPDATES):
        ids = torch.randint(len(x_train), (BATCH,), generator=g)
        xb, sb, yb = x_train[ids], src_train[ids], world["target_train"][ids]
        weights = torch.softmax(torch.einsum("bd,tdn->btn", xb, router), dim=-1)
        pred = torch.einsum("btn,bnd->btd", weights, sb)
        loss = torch.mean((pred - yb) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    elapsed = time.perf_counter() - start
    metrics: dict[str, object] = {}
    with torch.no_grad():
        for split, x, src in (("train", x_train, src_train),
                              ("validation", world["x_validation"], src_validation),
                              ("test", world["x_test"], src_test)):
            weights = torch.softmax(torch.einsum("bd,tdn->btn", x, router), dim=-1)
            pred = torch.einsum("btn,bnd->btd", weights, src)
            target = world[f"target_{split}"]
            per_target = [nrmse(pred[:, t], target[:, t]) for t in range(TARGETS)]
            metrics[f"fusion_{split}_nrmse_by_target"] = per_target
            metrics[f"fusion_{split}_nrmse"] = float(np.mean(per_target))
    return router.detach(), metrics, elapsed


def save_payload(path: Path, model: SourceBank, router: torch.Tensor) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **model.payload_arrays(router))
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def run(seed: int, condition: str, outdir: Path) -> dict[str, object]:
    world = make_world(seed)
    rows = []
    for method in SourceBank.METHODS:
        start = time.perf_counter()
        bank, src_metrics, bank_s = fit_bank(method, seed, world)
        router, fusion_metrics, fusion_s = fit_fusion(bank, seed, world)
        path = outdir / f"{condition}_{seed}_{method}.npz"
        size, digest = save_payload(path, bank, router)
        with torch.no_grad():
            x = world["x_test"]
            t0 = time.perf_counter()
            source = bank(x)
            weights = torch.softmax(torch.einsum("bd,tdn->btn", x, router), dim=-1)
            _ = torch.einsum("btn,bnd->btd", weights, source)
            infer_s = time.perf_counter() - t0
        rows.append({
            "method": method,
            "serialized_bytes": size,
            "payload_sha256": digest,
            "payload_path": path.name,
            **src_metrics,
            **fusion_metrics,
            "source_mac_proxy": __import__("model").mac_proxy(method),
            "target_fusion_mac_proxy": TARGETS * 2 * INPUT_DIM * SOURCES,
            "optimizer_updates": SOURCE_UPDATES + FUSION_UPDATES,
            "source_examples_seen": SOURCE_UPDATES * BATCH * SOURCES,
            "target_examples_seen": FUSION_UPDATES * BATCH * TARGETS,
            "source_training_wall_s": bank_s,
            "fusion_training_wall_s": fusion_s,
            "total_training_wall_s": time.perf_counter() - start,
            "inference_examples_per_s": len(x) / max(infer_s, 1e-9),
        })
    result = {
        "experiment_id": "MA-379", "condition": condition, "seed": seed,
        "source_updates": SOURCE_UPDATES, "fusion_updates_per_target": FUSION_UPDATES,
        "batch_size": BATCH, "learning_rate": LR, "results": rows,
    }
    return result


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
                      "results": [{k: r[k] for k in ("method", "serialized_bytes", "source_test_nrmse", "fusion_test_nrmse", "total_training_wall_s")}
                                  for r in result["results"]]}, indent=2))


if __name__ == "__main__":
    main()
