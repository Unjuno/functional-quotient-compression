from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from model import (Config, METHODS, MODES, Teacher, Top2MoE, active_mac_proxy,
                   coordinate_flop_proxy)


def make_inputs(seed: int, n: int, cfg: Config) -> torch.Tensor:
    return torch.randn((n, cfg.input_dim), generator=torch.Generator(device="cpu").manual_seed(seed))


def eval_model(model: Top2MoE, x: torch.Tensor, target: torch.Tensor,
               teacher_probs: torch.Tensor, teacher_indices: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        pred, logits, indices, _ = model(x)
        role_errors = []
        for role in range(model.cfg.experts):
            mask = (teacher_indices == role).any(dim=-1)
            role_errors.append(float(F.mse_loss(pred[mask], target[mask]).item()))
        agreement = (indices.unsqueeze(-1) == teacher_indices.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
        return {
            "heldout_mse": float(F.mse_loss(pred, target).item()),
            "top2_set_accuracy": float((agreement == model.cfg.top_k).float().mean().item()),
            "router_kl": float((teacher_probs * (teacher_probs.clamp_min(1e-12).log() - logits.log_softmax(-1))).sum(-1).mean().item()),
            "per_role_mse_json": json.dumps(role_errors, separators=(",", ":")),
        }


def train_one(world: int, teacher_seed: int, init_seed: int, mode: str, method: str,
              lr: float, updates: int, batch_size: int, inference_repetitions: int) -> dict:
    cfg = Config()
    torch.set_num_threads(1)
    teacher = Teacher(cfg, mode, teacher_seed)
    train_x = make_inputs(40_000_000 + world, updates * batch_size, cfg)
    val_x = make_inputs(50_000_000 + world, 2048, cfg)
    with torch.no_grad():
        train_y, train_p, train_i = teacher.forward(train_x)
        val_y, val_p, val_i = teacher.forward(val_x)
    torch.manual_seed(init_seed)
    model = Top2MoE(cfg, method)
    router_params = [p for n, p in model.named_parameters() if "router" in n]
    other_params = [p for n, p in model.named_parameters() if "router" not in n]
    opt = torch.optim.AdamW([
        {"params": router_params, "weight_decay": 0.0},
        {"params": other_params, "weight_decay": 1e-4},
    ], lr=lr)
    start = time.perf_counter()
    for step in range(updates):
        lo = step * batch_size
        xb, yb, pb = train_x[lo:lo + batch_size], train_y[lo:lo + batch_size], train_p[lo:lo + batch_size]
        model.train()
        pred, logits, _, _ = model(xb)
        loss = F.mse_loss(pred, yb) - (pb * logits.log_softmax(-1)).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    wall = time.perf_counter() - start
    metrics = eval_model(model, val_x, val_y, val_p, val_i)
    infer_x = make_inputs(60_000_000 + world, 1024, cfg)
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
    clone = Top2MoE(cfg, method)
    clone.load_state_dict(loaded["state_dict"])
    with torch.no_grad():
        if not torch.equal(clone(val_x)[0], model(val_x)[0]):
            raise RuntimeError("serialized model failed exact output round trip")
    return {
        "world": world, "teacher_seed": teacher_seed, "initialization_seed": init_seed,
        "teacher_mode": mode, "method": method, "learning_rate": lr,
        "updates": updates, "batch_size": batch_size, "examples_seen": updates * batch_size,
        **metrics, "serialized_bytes": len(payload),
        "active_mac_proxy_per_example": active_mac_proxy(cfg, method),
        "coordinate_flop_proxy_per_example": coordinate_flop_proxy(cfg, method),
        "training_wall_seconds": wall, "inference_batch_size": infer_x.shape[0],
        "inference_repetitions": inference_repetitions,
        "inference_examples_per_second": infer_x.shape[0] * inference_repetitions / inference_seconds,
        "parameter_count_diagnostic": sum(p.numel() for p in model.parameters()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
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
        runs = [(20000, 200000, 2000000, lr) for lr in (0.001, 0.003, 0.01)]
    else:
        if args.selection is None:
            ap.error("--selection is required for fresh")
        lr = float(json.loads(args.selection.read_text())["selected_learning_rate"])
        runs = [(w, 200000 + w - 20000, 2000000 + w - 20000, lr) for w in (20001, 20002, 20003)]
    rows = []
    for world, teacher_seed, init_seed, lr in runs:
        for mode in MODES:
            for idx, method in enumerate(METHODS):
                row = train_one(world, teacher_seed, init_seed + idx * 100, mode, method,
                                lr, args.updates, args.batch_size, args.inference_repetitions)
                print(f"{world} {mode} {method} lr={lr:g} mse={row['heldout_mse']:.7g} bytes={row['serialized_bytes']} t={row['training_wall_seconds']:.2f}s", flush=True)
                rows.append(row)
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
