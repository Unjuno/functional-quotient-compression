"""Deterministic development and fresh runner for MA-010."""
import argparse
import csv
import hashlib
import io
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from model import DIM, METHODS, ROLES, SequentialExperts

UPDATES = 1800
BATCH = 128


def rotation(angle, transpose=False):
    c, s = math.cos(angle), math.sin(angle)
    if transpose:
        s = -s
    r = torch.eye(DIM)
    r[0, 0], r[0, 1], r[1, 0], r[1, 1] = c, -s, s, c
    return r


def make_teacher(mode, seed):
    g = torch.Generator().manual_seed(seed)
    if mode == "aligned":
        base = torch.randn(DIM, DIM, generator=g) * 0.10
        base += torch.eye(DIM) * 0.18
        angles_in = torch.rand(ROLES, generator=g) * 1.4 - 0.7
        angles_out = torch.rand(ROLES, generator=g) * 1.4 - 0.7
        mats = torch.stack([
            rotation(float(angles_out[i])) @ base @ rotation(float(angles_in[i]), transpose=True)
            for i in range(ROLES)
        ])
    elif mode == "independent":
        mats = torch.randn(ROLES, DIM, DIM, generator=g) * 0.09
        mats += torch.stack([torch.eye(DIM) * (0.13 + 0.015 * i) for i in range(ROLES)])
    else:
        raise ValueError(mode)
    return mats


def make_data(mats, seed, n):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, DIM, generator=g)
    roles = torch.randint(ROLES, (n, 2), generator=g)
    y = apply_sequence(mats, x, roles)
    return x, roles, y


def apply_sequence(mats, x, roles):
    h = x
    for t in range(roles.shape[1]):
        h = torch.bmm(mats[roles[:, t]], h.unsqueeze(-1)).squeeze(-1)
    return h


def payload_bytes(model, method):
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, buf)
    config = json.dumps({"method": method, "dim": DIM, "roles": ROLES, "sequence_length": 2}, sort_keys=True, separators=(",", ":")).encode()
    return len(buf.getvalue()) + len(config)


def expected_mac(method):
    # Two dense 8x8 transforms; charge each method's extra learned transform work.
    base = 2 * DIM * DIM
    if method == "rank2":
        return base + 2 * (2 * DIM * 2)
    return base


def fit_eval(mode, world, teacher_seed, init_seed, method, lr, updates=UPDATES):
    teacher = make_teacher(mode, teacher_seed)
    x, roles, y = make_data(teacher, teacher_seed + 11, updates * BATCH)
    xe, re, ye = make_data(teacher, teacher_seed + 22, 8192)
    model = SequentialExperts(method, init_seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(world + 910000)
    picks = torch.randint(len(x), (updates, BATCH), generator=g)
    started = time.perf_counter()
    for step in range(updates):
        ix = picks[step]
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x[ix], roles[ix]), y[ix])
        loss.backward()
        opt.step()
    wall = time.perf_counter() - started
    with torch.no_grad():
        pred = model(xe, re)
        errors = (pred - ye).square().mean(dim=1)
        by_pair = {}
        pair_id = re[:, 0] * ROLES + re[:, 1]
        for p in range(ROLES * ROLES):
            mask = pair_id == p
            by_pair[f"{p // ROLES}>{p % ROLES}"] = float(errors[mask].mean())
        swaps = re.flip(dims=[1])
        swap_y = apply_sequence(teacher, xe, swaps)
        order_gap = float((ye - swap_y).square().mean())
        # Isolated eager throughput; every method receives identical role-pair inputs.
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            model(xe[:1024], re[:1024])
        infer_s = (time.perf_counter() - t0) / reps
    return {
        "world": world,
        "mode": mode,
        "method": method,
        "mse": float(errors.mean()),
        "max_pair_mse": max(by_pair.values()),
        "pair_mse": by_pair,
        "order_swap_mse_gap": order_gap,
        "bytes": payload_bytes(model, method),
        "examples": updates * BATCH,
        "updates": updates,
        "active_mac_proxy": expected_mac(method),
        "train_wall_seconds": wall,
        "inference_examples_per_second": 1024 / infer_s,
        "model": model,
    }


def save_rows(path, rows):
    keys = ["condition", "world_or_seed", "method", "serialized_bytes", "train_tokens_or_examples", "optimizer_updates", "active_compute_proxy", "wall_time_s", "primary_metric", "primary_value", "secondary_metric", "secondary_value", "status_note"]
    exists = Path(path).exists()
    with open(path, "a" if exists else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        if not exists:
            w.writeheader()
        for r in rows:
            note = {
                "max_pair_mse": r["max_pair_mse"],
                "pair_mse": r["pair_mse"],
                "order_swap_mse_gap": r["order_swap_mse_gap"],
                "inference_examples_per_second": r["inference_examples_per_second"],
                "teacher_seed": r["teacher_seed"],
                "initialization_seed": r["initialization_seed"],
                "learning_rate": r.get("lr"),
            }
            w.writerow({
                "condition": r["mode"], "world_or_seed": r["world"], "method": r["method"],
                "serialized_bytes": r["bytes"], "train_tokens_or_examples": r["examples"],
                "optimizer_updates": r["updates"], "active_compute_proxy": r["active_mac_proxy"],
                "wall_time_s": r["train_wall_seconds"], "primary_metric": "heldout_mse",
                "primary_value": r["mse"], "secondary_metric": "max_ordered_pair_mse",
                "secondary_value": r["max_pair_mse"], "status_note": json.dumps(note, sort_keys=True),
            })


def run_development(out_dir):
    lrs = [0.001, 0.003, 0.01]
    all_results = []
    for lr in lrs:
        for mode in ("aligned", "independent"):
            for mi, method in enumerate(METHODS):
                r = fit_eval(mode, 100000, 1000000 + (mode == "independent") * 100, 10000000 + mi * 1000, method, lr)
                r.update(teacher_seed=1000000 + (mode == "independent") * 100, initialization_seed=10000000 + mi * 1000)
                r["lr"] = lr
                all_results.append(r)
    # The loop above stores LR below; select using a pooled mean across all ten method/mode runs.
    by_lr = {lr: [] for lr in lrs}
    # rerun association is deterministic from iteration order (10 entries per candidate LR).
    for i, r in enumerate(all_results):
        lr = lrs[i // (len(METHODS) * 2)]
        r["lr"] = lr
        by_lr[lr].append(r["mse"])
    selected = min(lrs, key=lambda lr: sum(by_lr[lr]) / len(by_lr[lr]))
    out = Path(out_dir);out.mkdir(parents=True, exist_ok=True)
    save_rows(out / "RESULTS_CORE.csv", all_results)
    payload = [{k: v for k, v in r.items() if k != "model"} for r in all_results]
    (out / "dev_raw.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (out / "DEV_SELECTION.json").write_text(json.dumps({"selected_learning_rate": selected, "mean_heldout_mse_by_lr": {str(lr): sum(v) / len(v) for lr, v in by_lr.items()}, "development_world": 100000, "fresh_accessed": False}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected_learning_rate": selected, "mean_heldout_mse_by_lr": {str(lr): sum(v) / len(v) for lr, v in by_lr.items()}}, sort_keys=True))


def run_fresh(out_dir, lr):
    out = Path(out_dir);rows=[]
    for world, teacher_seed, init_seed in zip([100001,100002,100003],[1000001,1000002,1000003],[10000001,10000002,10000003]):
        for mode in ("aligned", "independent"):
            for mi, method in enumerate(METHODS):
                init=init_seed+mi*1000
                r=fit_eval(mode,world,teacher_seed+(mode=="independent")*100,init,method,lr)
                r.update(teacher_seed=teacher_seed+(mode=="independent")*100,initialization_seed=init)
                rows.append(r)
    save_rows(out/"RESULTS_CORE.csv",rows)
    serial=[{k:v for k,v in r.items() if k!="model"} for r in rows]
    (out/"fresh_raw.json").write_text(json.dumps(serial,indent=2,sort_keys=True)+"\n")
    # Payload round-trip bytes are audited from the fitted state dict separately by the verifier.
    print(json.dumps({"fresh_rows":len(rows),"selected_learning_rate":lr,"aligned_mirror_vs_full":{str(w):next(r["mse"] for r in rows if r["world"]==w and r["mode"]=="aligned" and r["method"]=="mirror")/next(r["mse"] for r in rows if r["world"]==w and r["mode"]=="aligned" and r["method"]=="independent") for w in [100001,100002,100003]}},sort_keys=True))


def main():
    torch.set_num_threads(1)
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=["development","fresh"])
    p.add_argument("--out",required=True)
    p.add_argument("--lr",type=float)
    a=p.parse_args()
    if a.phase=="development": run_development(a.out)
    else:
        if a.lr is None: raise SystemExit("--lr required for fresh")
        run_fresh(a.out,a.lr)


if __name__=="__main__":main()
