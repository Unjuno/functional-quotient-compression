#!/usr/bin/env python3
"""Development/fresh runner for the MA-241 mechanism screen."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import Config, MoEViews, Teacher, mac_proxy_per_example

METHODS = ("untied", "tied", "gate", "lowrank", "mirror")
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "RESULTS_CORE.csv"
FIELDS = ["condition", "world_or_seed", "method", "serialized_bytes", "train_tokens_or_examples", "optimizer_updates", "active_compute_proxy", "wall_time_s", "primary_metric", "primary_value", "secondary_metric", "secondary_value", "status_note"]


def build_data(cfg: Config, world: int, split_seed: int, init_seed: int):
    teacher = Teacher(cfg, world)
    g = torch.Generator().manual_seed(split_seed)
    def sample(n: int):
        x = torch.randn(n, cfg.input_dim, generator=g)
        layer = torch.arange(n) % cfg.layers
        perm = torch.randperm(n, generator=g)
        x, layer = x[perm], layer[perm]
        with torch.no_grad():
            y = teacher(x, layer)
        return x, layer, y
    train, dev, test = sample(8192), sample(2048), sample(4096)
    return train, dev, test


def mse(model, data):
    x, layer, y = data
    model.eval()
    with torch.no_grad():
        return float(torch.mean((model(x, layer) - y) ** 2))


def fit(cfg: Config, method: str, data, init_seed: int, lr: float, updates: int):
    torch.manual_seed(init_seed)
    model = MoEViews(cfg, method)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x, layer, y = data
    generator = torch.Generator().manual_seed(init_seed + 17)
    model.train()
    t0 = time.perf_counter()
    for _ in range(updates):
        indices = torch.randint(len(x), (128,), generator=generator)
        pred = model(x[indices], layer[indices])
        loss = torch.mean((pred - y[indices]) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - t0
    return model, elapsed


def serialize_bytes(model: MoEViews) -> int:
    # inference_payload_bytes includes the serialized state_dict and paid config metadata.
    return model.inference_payload_bytes()


def row(condition, world, method, model, elapsed, updates, value, secondary, status):
    examples = updates * 128
    macs = mac_proxy_per_example(model.cfg, method) * examples * 3  # forward + backward proxy
    return {
        "condition": condition,
        "world_or_seed": world,
        "method": method,
        "serialized_bytes": serialize_bytes(model),
        "train_tokens_or_examples": examples,
        "optimizer_updates": updates,
        "active_compute_proxy": macs,
        "wall_time_s": round(elapsed, 6),
        "primary_metric": "heldout_mse",
        "primary_value": f"{value:.9g}",
        "secondary_metric": secondary[0],
        "secondary_value": secondary[1],
        "status_note": status,
    }


def append(rows):
    exists = RESULTS.exists()
    with RESULTS.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("dev", "fresh"), required=True)
    p.add_argument("--lr", type=float)
    args = p.parse_args()
    torch.set_num_threads(1)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    cfg = Config(residual_rank=1)
    updates = 1800
    if args.phase == "dev":
        worlds = [(24100, 241000, 2410000)]
        lrs = [1e-3, 3e-3]
    else:
        if args.lr not in (1e-3, 3e-3):
            raise SystemExit("fresh requires frozen --lr 0.001 or --lr 0.003")
        worlds = [(24101, 241010, 2410100), (24102, 241020, 2410200), (24103, 241030, 2410300)]
        lrs = [args.lr]
    all_rows = []
    dev_scores = {lr: [] for lr in lrs}
    for world, split_seed, init_seed in worlds:
        train, dev, test = build_data(cfg, world, split_seed, init_seed)
        for lr in lrs:
            for method in METHODS:
                model, elapsed = fit(cfg, method, train, init_seed + METHODS.index(method) * 100 + int(lr * 100000), lr, updates)
                valid = mse(model, dev)
                if args.phase == "dev":
                    dev_scores[lr].append(valid)
                    secondary = ("dev_mse", f"{valid:.9g}")
                    note = f"lr_candidate={lr}; development split only"
                    all_rows.append(row("dev", world, method + f"_lr{lr:g}", model, elapsed, updates, valid, secondary, note))
                else:
                    test_value = mse(model, test)
                    all_rows.append(row("fresh", world, method, model, elapsed, updates, test_value, ("parameters", str(model.parameter_count())), f"frozen_lr={lr}; cpu_threads=1"))
                print(f"{args.phase} world={world} method={method} lr={lr:g} dev={valid:.6f}" + (f" test={test_value:.6f}" if args.phase == "fresh" else "") + f" bytes={serialize_bytes(model)} time={elapsed:.2f}s", flush=True)
    if args.phase == "dev":
        selected = min(dev_scores, key=lambda lr: float(np.mean(dev_scores[lr])))
        freeze = {
            "experiment_id": "MA-241", "selected_common_lr": selected,
            "selection": "lowest mean development heldout MSE across all five methods",
            "development_mse_by_lr": {str(lr): [float(x) for x in scores] for lr, scores in dev_scores.items()},
            "fresh_worlds": [24101, 24102, 24103], "updates": updates,
            "source_freeze_stage": "before fresh data generation/evaluation"
        }
        (ROOT / "DEV_SELECTION.json").write_text(json.dumps(freeze, indent=2) + "\n")
        print("selected_common_lr=", selected, flush=True)
    append(all_rows)
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "threads": torch.get_num_threads(), "rows": len(all_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
