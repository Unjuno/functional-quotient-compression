from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from model import (Config, METHODS, MODES, ExpertChoiceMoE, Teacher,
                   active_mac_proxy, coordinate_flop_proxy)


ROLE_CENTERS = torch.tensor([[1.4, 1.4], [-1.4, 1.4], [-1.4, -1.4], [1.4, -1.4]])


def make_balanced(seed: int, n: int, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    if n % cfg.experts:
        raise ValueError("balanced synthetic set must divide evenly among roles")
    # Interleave each batch's role groups to guarantee exact per-batch capacity.
    bsz = cfg.capacity_per_expert * cfg.experts
    if n % bsz:
        raise ValueError("balanced set must be an integer number of capacity batches")
    labels = torch.arange(cfg.experts).repeat_interleave(cfg.capacity_per_expert).repeat(n // bsz)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((n, cfg.input_dim), generator=gen)
    x[:, :2] += ROLE_CENTERS[labels]
    return x, labels


def eval_model(model, x, labels, target):
    model.eval()
    with torch.no_grad():
        pred, logits, mask, weights = model(x)
        role_recall = mask.gather(1, labels[:, None]).float().mean().item()
        role_mse = []
        for role in range(model.cfg.experts):
            selected = labels == role
            role_mse.append(float(F.mse_loss(pred[selected], target[selected]).item()))
        covered = mask.any(dim=-1)
        calls = covered.float().sum() if model.method == "tied_ec" else mask.float().sum()
        mean_assignments = float(calls.item() / x.shape[0])
        return {
            "heldout_mse": float(F.mse_loss(pred, target).item()),
            "router_accuracy": float((logits.argmax(-1) == labels).float().mean().item()),
            "role_recall": float(role_recall),
            "token_coverage": float(covered.float().mean().item()),
            "mean_expert_assignments_per_token": float(mask.float().sum().item() / x.shape[0]),
            "active_ffn_calls_per_token": mean_assignments,
            "expert_loads_json": json.dumps(mask.sum(dim=0).tolist(), separators=(",", ":")),
            "per_role_mse_json": json.dumps(role_mse, separators=(",", ":")),
            "active_mac_proxy_per_example": active_mac_proxy(model.cfg, model.method, mean_assignments),
            "coordinate_flop_proxy_per_example": coordinate_flop_proxy(model.cfg, model.method, mean_assignments),
        }


def train_one(world: int, teacher_seed: int, init_seed: int, mode: str, method: str,
              lr: float, updates: int, batch_size: int, inference_repetitions: int) -> dict:
    cfg = Config()
    torch.set_num_threads(1)
    if batch_size != cfg.experts * cfg.capacity_per_expert:
        raise ValueError("protocol batch size must match per-expert expert-choice capacity")
    teacher = Teacher(cfg, mode, teacher_seed)
    train_x, train_labels = make_balanced(70_000_000 + world, updates * batch_size, cfg)
    val_x, val_labels = make_balanced(80_000_000 + world, 2048, cfg)
    with torch.no_grad():
        train_y = teacher.forward(train_x, train_labels)
        val_y = teacher.forward(val_x, val_labels)
    torch.manual_seed(init_seed)
    model = ExpertChoiceMoE(cfg, method)
    router_params = [p for n, p in model.named_parameters() if "router" in n]
    other_params = [p for n, p in model.named_parameters() if "router" not in n]
    opt = torch.optim.AdamW([
        {"params": router_params, "weight_decay": 0.0},
        {"params": other_params, "weight_decay": 1e-4},
    ], lr=lr)
    start = time.perf_counter()
    for step in range(updates):
        lo = step * batch_size
        xb = train_x[lo:lo + batch_size]
        yb = train_y[lo:lo + batch_size]
        labels = train_labels[lo:lo + batch_size]
        model.train()
        pred, logits, _, _ = model(xb)
        loss = F.mse_loss(pred, yb) + F.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    wall = time.perf_counter() - start
    metrics = eval_model(model, val_x, val_labels, val_y)
    infer_x, infer_labels = make_balanced(90_000_000 + world, 1024, cfg)
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(infer_x)
        t0 = time.perf_counter()
        for _ in range(inference_repetitions):
            model(infer_x)
        inference_seconds = time.perf_counter() - t0
    payload = model.serialize()
    loaded = torch.load(__import__("io").BytesIO(payload), map_location="cpu", weights_only=False)
    clone = ExpertChoiceMoE(cfg, method)
    clone.load_state_dict(loaded["state_dict"])
    with torch.no_grad():
        if not torch.equal(clone(val_x)[0], model(val_x)[0]):
            raise RuntimeError("serialized payload failed exact output round trip")
    return {
        "world": world, "teacher_seed": teacher_seed, "initialization_seed": init_seed,
        "teacher_mode": mode, "method": method, "learning_rate": lr,
        "updates": updates, "batch_size": batch_size, "examples_seen": updates * batch_size,
        **metrics, "serialized_bytes": len(payload), "training_wall_seconds": wall,
        "inference_batch_size": infer_x.shape[0], "inference_repetitions": inference_repetitions,
        "inference_examples_per_second": infer_x.shape[0] * inference_repetitions / inference_seconds,
        "parameter_count_diagnostic": sum(p.numel() for p in model.parameters()),
    }


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("dev", "fresh"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--selection", type=Path)
    ap.add_argument("--updates", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--inference-repetitions", type=int, default=20)
    args = ap.parse_args()
    if args.stage == "dev":
        runs = [(60000, 600000, 6000000, lr) for lr in (0.001, 0.003, 0.01)]
    else:
        if args.selection is None:
            ap.error("--selection is required for fresh")
        lr = float(json.loads(args.selection.read_text())["selected_learning_rate"])
        runs = [(w, 600000 + w - 60000, 6000000 + w - 60000, lr) for w in (60001, 60002, 60003)]
    rows = []
    for world, teacher_seed, init_seed, lr in runs:
        for mode in MODES:
            for idx, method in enumerate(METHODS):
                row = train_one(world, teacher_seed, init_seed + idx * 100, mode, method,
                                lr, args.updates, args.batch_size, args.inference_repetitions)
                print(f"{world} {mode} {method} lr={lr:g} mse={row['heldout_mse']:.7g} bytes={row['serialized_bytes']} cov={row['token_coverage']:.3f} t={row['training_wall_seconds']:.2f}s", flush=True)
                rows.append(row)
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
