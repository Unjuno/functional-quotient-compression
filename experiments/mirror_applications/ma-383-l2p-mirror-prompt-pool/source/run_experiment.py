#!/usr/bin/env python3
"""Synthetic L2P prompt pool experiment with explicit/scalar/Mirror/hyper controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def dataset(seed: int, family: str, split: str):
    g = torch.Generator().manual_seed(seed + {"train": 0, "validation": 1000, "test": 2000}[split])
    slots, dim, classes = 8, 32, 4
    # Domain identity is observable from the query, but never passed as an ID.
    centers = torch.zeros(slots, dim)
    centers[torch.arange(slots), torch.arange(slots)] = 3.0
    wgen = torch.Generator().manual_seed(seed + 991)
    w = torch.randn(dim, classes, generator=wgen) / math.sqrt(dim)
    pgen = torch.Generator().manual_seed(seed + 1771)
    base = torch.randn(dim, generator=pgen) * 0.55
    if family == "aligned":
        angles = torch.linspace(-0.9, 0.9, slots)
        prompts = torch.stack([rotate(base, float(a)) for a in angles])
    else:
        prompts = torch.randn(slots, dim, generator=pgen) * 0.55
    n = {"train": 1024, "validation": 256, "test": 512}[split]
    xs, ys, ids = [], [], []
    for t in range(slots):
        x = centers[t] + 0.50 * torch.randn(n, dim, generator=g)
        y = ((x + prompts[t]) @ w).argmax(-1)
        xs.append(x); ys.append(y); ids.append(torch.full((n,), t, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys), torch.cat(ids), centers, w, prompts


def rotate(v: torch.Tensor, theta: float):
    # Shared adjacent-pair Givens view, identical operation for all slots.
    out = v.clone()
    if isinstance(theta, torch.Tensor):
        c, s = torch.cos(theta), torch.sin(theta)
    else:
        c, s = math.cos(theta), math.sin(theta)
    a, b = v[0::2], v[1::2]
    out[0::2] = c * a - s * b
    out[1::2] = s * a + c * b
    return out


class Pool(nn.Module):
    def __init__(self, method: str, slots=8, dim=32, classes=4):
        super().__init__(); self.method = method
        if method == "explicit": self.prompt = nn.Parameter(torch.zeros(slots, dim))
        elif method in ("scalar", "mirror"):
            self.basis = nn.Parameter(torch.zeros(dim))
            self.code = nn.Parameter(torch.zeros(slots))
        elif method == "hyper":
            self.embed = nn.Embedding(slots, 8)
            self.gen = nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, dim))
        else: raise ValueError(method)

    def prompts(self):
        if self.method == "explicit": return self.prompt
        if self.method == "scalar": return self.code[:, None] * self.basis[None, :]
        if self.method == "mirror": return torch.stack([rotate(self.basis, a) for a in self.code])
        idx = torch.arange(8, device=self.embed.weight.device)
        return self.gen(self.embed(idx))


def choose_keys(x, keys):
    # Euclidean nearest key; the task label is not an input to inference.
    return torch.cdist(x.float(), keys.float()).argmin(-1)


def archive(path: Path, method, model, keys, w):
    state = {k: v.detach().cpu().numpy().astype(np.float16) for k, v in model.state_dict().items()}
    state.update({"keys": keys.cpu().numpy().astype(np.float16), "classifier": w.cpu().numpy().astype(np.float16)})
    meta = json.dumps({"method": method, "slot_count": 8, "dim": 32}, sort_keys=True).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("metadata.json", meta)
        for k, a in sorted(state.items()):
            b = Path("arrays") / (k.replace(".", "_") + ".npy")
            import io
            buf = io.BytesIO(); np.save(buf, a, allow_pickle=False); z.writestr(str(b), buf.getvalue())
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


@torch.no_grad()
def evaluate(model, x, y, task, keys, w):
    model.eval(); slots = choose_keys(x, keys)
    prompt = model.prompts()[slots]
    logits = (x + prompt) @ w
    acc = (logits.argmax(-1) == y).float()
    per = [float(acc[task == t].mean()) for t in range(8)]
    return {"accuracy": float(acc.mean()), "per_task_accuracy": per,
            "retrieval_accuracy": float((slots == task).float().mean()),
            "nll": float(nn.functional.cross_entropy(logits, y)),
            "retrieved_slots": slots}


def train_one(seed, family, method, updates, device):
    x, y, task, keys, w, target_prompts = dataset(seed, family, "train")
    xv, yv, tv, _, _, _ = dataset(seed, family, "validation")
    xt, yt, tt, _, _, _ = dataset(seed, family, "test")
    model = Pool(method).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    x, y, task = x.to(device), y.to(device), task.to(device)
    start = time.perf_counter()
    for step in range(updates):
        idx = torch.randint(len(x), (128,), device=device)
        slots = choose_keys(x[idx], keys.to(device))
        logits = (x[idx] + model.prompts()[slots]) @ w.to(device)
        loss = nn.functional.cross_entropy(logits, y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    train_wall = time.perf_counter() - start
    keys = keys.to(device); w = w.to(device)
    val = evaluate(model, xv.to(device), yv.to(device), tv.to(device), keys, w)
    test = evaluate(model, xt.to(device), yt.to(device), tt.to(device), keys, w)
    # Generation MAC proxy excludes shared matrix multiplication and retrieval.
    if method == "explicit": gen_macs = 0
    elif method in ("scalar", "mirror"): gen_macs = 32 * 8
    else: gen_macs = (8*8*32 + 8*32*32 + 8*32*32)
    return model, keys, w, {"validation": {k:v for k,v in val.items() if k != "retrieved_slots"},
        "test": {k:v for k,v in test.items() if k != "retrieved_slots"},
        "train_wall_seconds": train_wall, "updates": updates, "examples_seen": updates*128,
        "retrieval_distance_ops_per_query": 8*32, "prompt_generation_mac_proxy_per_query": gen_macs,
        "theoretical_payload_parameters": sum(p.numel() for p in model.parameters()) + keys.numel() + w.numel()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--family", choices=["aligned", "unrelated"], default="aligned")
    ap.add_argument("--updates", type=int, default=1600); ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True); args = ap.parse_args()
    result = {"experiment_id":"MA-383", "seed":args.seed, "family":args.family, "methods":{}}
    for method in ["explicit", "scalar", "mirror", "hyper"]:
        model, keys, w, row = train_one(args.seed,args.family,method,args.updates,args.device)
        dest = args.out / f"{args.family}_{args.seed}_{method}.zip"
        size, digest = archive(dest, method, model, keys, w)
        row["actual_payload_bytes"] = size; row["payload_sha256"] = digest
        result["methods"][method] = row
        del model
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/f"{args.family}_{args.seed}_result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__ == "__main__": main()
