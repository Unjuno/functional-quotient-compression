#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, platform, time
from pathlib import Path
import torch
from model import METHODS, SignedExperts, compute_proxy, givens, signed_coefficients

ROOT = Path(__file__).resolve().parents[1]
D, O, N, UPDATES, BATCH = 16, 12, 4, 1200, 64
FIELDS = ["condition", "world_or_seed", "mode", "method", "serialized_model_bytes", "train_examples", "optimizer_updates", "active_compute_proxy", "wall_time_s", "inference_examples_per_s", "primary_metric", "primary_value", "secondary_metric", "secondary_value", "status_note"]


def make_world(seed: int, mode: str):
    g = torch.Generator().manual_seed(seed)
    if mode == "aligned_views":
        return {"weight": torch.randn(D, O, generator=g) * 0.6,
                "angles": torch.rand(N, D // 2, generator=g) * 1.1 - 0.55,
                "independent": None}
    return {"weight": None, "angles": None,
            "independent": torch.randn(N, D, O, generator=g) * 0.6}


def target(x: torch.Tensor, world, mode: str) -> torch.Tensor:
    c = signed_coefficients(x)
    if mode == "aligned_views":
        views = torch.stack([givens(x, a.expand(x.shape[0], -1)) for a in world["angles"]], dim=1)
        components = torch.einsum("bnd,do->bno", views, world["weight"])
    else:
        components = torch.einsum("bd,ndo->bno", x, world["independent"])
    return torch.einsum("bn,bno->bo", c, components)


def fit(method: str, world, mode: str, init_seed: int, data_seed: int, lr: float):
    torch.manual_seed(init_seed)
    model = SignedExperts(method, init_seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(data_seed + 17)
    start = time.perf_counter()
    model.train()
    for _ in range(UPDATES):
        x = torch.randn(BATCH, D, generator=g)
        y = target(x, world, mode)
        loss = (model(x) - y).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model, time.perf_counter() - start


def evaluate(model, world, mode: str, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4096, D, generator=g)
    y = target(x, world, mode)
    model.eval()
    with torch.no_grad():
        pred = model(x)
        mse = float((pred - y).square().mean())
        r2 = float(1 - (pred - y).square().sum() / (y - y.mean()).square().sum())
        per_pattern = []
        # Walsh code is uniquely determined by the two signs; audit all four cells.
        for pattern in range(N):
            a = (x[:, 0] >= 0); b = (x[:, 1] >= 0)
            bit = a.long() * 2 + b.long()
            selected = bit == pattern
            per_pattern.append(float((pred[selected] - y[selected]).square().mean()))
    return mse, r2, max(per_pattern), x


def throughput(model, x):
    x = x[:64]
    model.eval()
    with torch.no_grad():
        for _ in range(5): model(x)
        start = time.perf_counter()
        for _ in range(30): model(x)
    return 64 * 30 / (time.perf_counter() - start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "fresh"], required=True)
    ap.add_argument("--lr", type=float)
    args = ap.parse_args()
    torch.set_num_threads(1)
    modes = ["aligned_views", "independent_experts"]
    if args.phase == "dev":
        worlds, lrs = [(50000, 500000)], [0.003, 0.01]
    else:
        if args.lr not in (0.003, 0.01): raise SystemExit("fresh requires a frozen development LR")
        worlds, lrs = [(50001, 500001), (50002, 500002), (50003, 500003)], [args.lr]
    rows, pooled = [], {lr: [] for lr in lrs}
    for world_id, init in worlds:
        for mode in modes:
            world = make_world(world_id, mode)
            data_seed = init + (0 if mode == modes[0] else 10000)
            for lr in lrs:
                for ix, method in enumerate(METHODS):
                    model, elapsed = fit(method, world, mode, init + ix * 193 + int(lr * 10000), data_seed, lr)
                    mse, r2, worst, x = evaluate(model, world, mode, init + 91)
                    if args.phase == "dev": pooled[lr].append(mse)
                    row = {"condition": args.phase, "world_or_seed": world_id, "mode": mode, "method": method,
                           "serialized_model_bytes": model.serialized_payload_bytes(), "train_examples": UPDATES * BATCH,
                           "optimizer_updates": UPDATES, "active_compute_proxy": compute_proxy(method, UPDATES * BATCH),
                           "wall_time_s": round(elapsed, 6), "inference_examples_per_s": round(throughput(model, x), 3),
                           "primary_metric": "signed_mixture_MSE", "primary_value": f"{mse:.10g}",
                           "secondary_metric": "R2;worst_sign_pattern_MSE", "secondary_value": f"{r2:.8g};{worst:.10g}",
                           "status_note": f"lr={lr}; matched minibatches; signed Walsh coefficients; no learned router"}
                    rows.append(row)
                    print(args.phase, world_id, mode, method, lr, "MSE", mse, "worst_pattern", worst, "bytes", row["serialized_model_bytes"], flush=True)
    if args.phase == "dev":
        best = min(pooled, key=lambda lr: sum(pooled[lr]) / len(pooled[lr]))
        (ROOT / "DEV_SELECTION.json").write_text(json.dumps({"experiment_id":"MA-005","selected_common_lr":best,
            "mean_mse_by_lr":{str(k):sum(v)/len(v) for k,v in pooled.items()},"fresh_worlds":[50001,50002,50003],
            "updates":UPDATES,"development_generation":"v1"}, indent=2)+"\n")
    with (ROOT / "RESULTS_CORE.csv").open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n").writerows(rows)
    print(json.dumps({"python":platform.python_version(),"torch":torch.__version__,"threads":torch.get_num_threads(),"rows":len(rows)}))

if __name__ == "__main__": main()
