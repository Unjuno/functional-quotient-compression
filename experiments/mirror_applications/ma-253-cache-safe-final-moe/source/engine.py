#!/usr/bin/env python3
"""Train and evaluate MA-253 synthetic domain-update controls."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import torch
from model import Config, DomainUpdate, mac_proxy_per_example

ROOT = Path(__file__).resolve().parents[1]
METHODS = DomainUpdate.METHODS
FIELDS = ["condition", "world_or_seed", "method", "serialized_bytes", "train_tokens_or_examples", "optimizer_updates", "active_compute_proxy", "wall_time_s", "primary_metric", "primary_value", "secondary_metric", "secondary_value", "status_note"]


def teacher_and_data(cfg: Config, world: int, init_seed: int):
    g = torch.Generator().manual_seed(world)
    base = torch.randn(cfg.input_dim, cfg.output_dim, generator=g) * 0.08
    a = torch.randn(cfg.domains, cfg.input_dim, cfg.rank, generator=g) * 0.16
    b = torch.randn(cfg.domains, cfg.rank, cfg.output_dim, generator=g) * 0.16
    data_gen = torch.Generator().manual_seed(init_seed)
    def sample(n):
        x = torch.randn(n, cfg.input_dim, generator=data_gen)
        domain = torch.arange(n) % cfg.domains
        order = torch.randperm(n, generator=data_gen)
        x, domain = x[order], domain[order]
        y = x @ base + torch.einsum("bd,bdr,bro->bo", x, a[domain], b[domain])
        return x, domain, y
    return base, sample(8192), sample(2048), sample(4096)


def mse(model, data):
    model.eval()
    with torch.no_grad():
        return float(torch.mean((model(data[0], data[1]) - data[2]) ** 2))


def fit(cfg, method, base, train, init_seed, lr, updates):
    torch.manual_seed(init_seed)
    model = DomainUpdate(cfg, method, base)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x, domain, target = train
    g = torch.Generator().manual_seed(init_seed + 101)
    t0 = time.perf_counter()
    model.train()
    for _ in range(updates):
        ix = torch.randint(len(x), (128,), generator=g)
        pred = model(x[ix], domain[ix])
        loss = torch.mean((pred - target[ix]) ** 2)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return model, time.perf_counter() - t0


def make_row(phase, world, method, model, elapsed, updates, value, secondary, lr):
    examples = updates * 128
    macs = mac_proxy_per_example(model.cfg, method) * examples * 3
    return {"condition": phase, "world_or_seed": world, "method": method,
            "serialized_bytes": model.inference_payload_bytes(), "train_tokens_or_examples": examples,
            "optimizer_updates": updates, "active_compute_proxy": macs, "wall_time_s": round(elapsed, 6),
            "primary_metric": "heldout_mse", "primary_value": f"{value:.10g}",
            "secondary_metric": secondary[0], "secondary_value": secondary[1],
            "status_note": f"lr={lr}; domain address supplied to all methods"}


def append_rows(rows):
    path = ROOT / "RESULTS_CORE.csv"
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        if f.tell() == 0: w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("dev", "fresh"), required=True)
    p.add_argument("--lr", type=float)
    args = p.parse_args()
    torch.set_num_threads(1)
    cfg = Config()
    updates = 1500
    if args.phase == "dev":
        worlds, lrs = [(25300, 253000)], [0.003, 0.01]
    else:
        if args.lr not in (0.003, 0.01): raise SystemExit("fresh requires frozen --lr 0.003 or 0.01")
        worlds, lrs = [(25301,253001),(25302,253002),(25303,253003)], [args.lr]
    rows=[]; dev_scores={lr:[] for lr in lrs}
    for world, init_seed in worlds:
        base, train, dev, test = teacher_and_data(cfg, world, init_seed)
        for lr in lrs:
            for i,method in enumerate(METHODS):
                model,elapsed=fit(cfg,method,base,train,init_seed+i*137+int(lr*10000),lr,updates)
                valid=mse(model,dev)
                if args.phase=="dev":
                    dev_scores[lr].append(valid)
                    rows.append(make_row("dev",world,method+f"_lr{lr:g}",model,elapsed,updates,valid,("dev_mse",f"{valid:.10g}"),lr))
                else:
                    test_value=mse(model,test)
                    rows.append(make_row("fresh",world,method,model,elapsed,updates,test_value,("parameters",str(model.parameter_count())),lr))
                print(f"{args.phase} world={world} method={method} lr={lr} dev={valid:.7g}" + (f" test={test_value:.7g}" if args.phase=="fresh" else "") + f" bytes={model.inference_payload_bytes()} wall={elapsed:.2f}s",flush=True)
    if args.phase=="dev":
        selected=min(dev_scores,key=lambda lr:sum(dev_scores[lr])/len(dev_scores[lr]))
        freeze={"experiment_id":"MA-253","selected_common_lr":selected,"dev_mse_by_lr":{str(k):v for k,v in dev_scores.items()},"fresh_worlds":[25301,25302,25303],"updates":updates,"source_freeze_stage":"before fresh evaluation"}
        (ROOT/"DEV_SELECTION.json").write_text(json.dumps(freeze,indent=2)+"\n")
        print("selected_common_lr=",selected)
    append_rows(rows)
    print(json.dumps({"python":platform.python_version(),"torch":torch.__version__,"threads":torch.get_num_threads(),"rows":len(rows)}))


if __name__=="__main__": main()
