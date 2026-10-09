from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from model import Config, METHODS, MODES, Teacher, Top1MoE, active_mac_proxy, coordinate_flop_proxy, route_labels


def make_inputs(seed: int, n: int, cfg: Config) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((n, cfg.input_dim), generator=generator)


def eval_model(model: Top1MoE, x: torch.Tensor, labels: torch.Tensor, target: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        pred, logits, route = model(x)
        per_role = []
        for r in range(model.cfg.experts):
            mask = labels == r
            per_role.append(float(F.mse_loss(pred[mask], target[mask]).item()))
        return {
            "heldout_mse": float(F.mse_loss(pred, target).item()),
            "router_accuracy": float((route == labels).float().mean().item()),
            "router_ce": float(F.cross_entropy(logits, labels).item()),
            "per_role_mse_json": json.dumps(per_role, separators=(",", ":")),
        }


def train_one(world: int, teacher_seed: int, init_seed: int, mode: str, method: str, lr: float,
              updates: int, batch_size: int, inference_repetitions: int) -> dict:
    cfg = Config()
    torch.set_num_threads(1)
    teacher = Teacher(cfg, mode, teacher_seed)
    # Identical train/validation examples within world and teacher mode for every method/LR.
    train_x = make_inputs(20_000_000 + world, updates * batch_size, cfg)
    train_y = route_labels(train_x)
    with torch.no_grad():
        train_t = teacher.forward(train_x, train_y)
    val_x = make_inputs(30_000_000 + world, 2048, cfg)
    val_y = route_labels(val_x)
    with torch.no_grad():
        val_t = teacher.forward(val_x, val_y)
    torch.manual_seed(init_seed)
    model = Top1MoE(cfg, method)
    router_params = [p for n, p in model.named_parameters() if "router" in n]
    other_params = [p for n, p in model.named_parameters() if "router" not in n]
    opt = torch.optim.AdamW([
        {"params": router_params, "weight_decay": 0.0},
        {"params": other_params, "weight_decay": 1e-4},
    ], lr=lr)
    start = time.perf_counter()
    for step in range(updates):
        lo = step * batch_size
        xb, yb, tb = train_x[lo:lo + batch_size], train_y[lo:lo + batch_size], train_t[lo:lo + batch_size]
        model.train()
        pred, logits, _ = model(xb)
        loss = F.mse_loss(pred, tb) + F.cross_entropy(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    train_seconds = time.perf_counter() - start
    metrics = eval_model(model, val_x, val_y, val_t)
    model.eval()
    inference_x = make_inputs(40_000_000 + world, 1024, cfg)
    with torch.no_grad():
        for _ in range(3):
            model(inference_x)
        t0 = time.perf_counter()
        for _ in range(inference_repetitions):
            model(inference_x)
        inference_seconds = time.perf_counter() - t0
    payload = model.serialize()
    serialized_bytes = len(payload)
    # Exercise exact load/reconstruction and check payload round-trip state.
    loaded = torch.load(__import__("io").BytesIO(payload), map_location="cpu", weights_only=False)
    clone = Top1MoE(cfg, method)
    clone.load_state_dict(loaded["state_dict"])
    with torch.no_grad():
        replay = clone(val_x)[0]
    if not torch.equal(replay, model(val_x)[0]):
        raise RuntimeError("serialized payload did not reproduce model outputs exactly")
    return {
        "world": world,
        "teacher_seed": teacher_seed,
        "initialization_seed": init_seed,
        "teacher_mode": mode,
        "method": method,
        "learning_rate": lr,
        "updates": updates,
        "batch_size": batch_size,
        "examples_seen": updates * batch_size,
        **metrics,
        "serialized_bytes": serialized_bytes,
        "active_mac_proxy_per_example": active_mac_proxy(cfg, method),
        "coordinate_flop_proxy_per_example": coordinate_flop_proxy(cfg, method),
        "training_wall_seconds": train_seconds,
        "inference_batch_size": inference_x.shape[0],
        "inference_repetitions": inference_repetitions,
        "inference_examples_per_second": inference_x.shape[0] * inference_repetitions / inference_seconds,
        "parameter_count_diagnostic": sum(p.numel() for p in model.parameters()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("dev", "fresh"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--selection", type=Path)
    ap.add_argument("--updates", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--inference-repetitions", type=int, default=20)
    args = ap.parse_args()
    if args.stage == "dev":
        runs = [(10000, 100000, 1000000, lr) for lr in (0.001, 0.003, 0.01)]
    else:
        if args.selection is None:
            ap.error("--selection is required for fresh")
        selection = json.loads(args.selection.read_text())
        lr = float(selection["selected_learning_rate"])
        runs = [(w, 100000 + (w - 10000), 1000000 + (w - 10000), lr) for w in (10001, 10002, 10003)]
    rows: list[dict] = []
    for world, teacher_seed, init_seed, lr in runs:
        for mode in MODES:
            for idx, method in enumerate(METHODS):
                # Stable, method-specific initialization; data/teacher seeds remain matched.
                method_init = init_seed + idx * 100
                row = train_one(world, teacher_seed, method_init, mode, method, lr,
                                args.updates, args.batch_size, args.inference_repetitions)
                print(f"{world} {mode} {method} lr={lr:g} mse={row['heldout_mse']:.7g} bytes={row['serialized_bytes']} t={row['training_wall_seconds']:.2f}s", flush=True)
                rows.append(row)
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
