#!/usr/bin/env python3
"""Reproducible simulated-client screen for MA-341."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
METHODS = ("pfedhn", "mirror", "film", "global", "independent")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def partition(seed: int):
    """Per-class Dirichlet assignment, then a persisted within-client split."""
    rng = np.random.default_rng(seed)
    ds = load_digits()
    x = ds.data.astype(np.float32) / 16.0
    y = ds.target.astype(np.int64)
    clients = [[] for _ in range(12)]
    for label in range(10):
        ids = np.flatnonzero(y == label)
        rng.shuffle(ids)
        proportions = rng.dirichlet(np.full(12, 0.5))
        assignment = rng.choice(12, size=len(ids), p=proportions)
        for i, c in zip(ids, assignment):
            clients[int(c)].append(int(i))
    splits = []
    for c, ids in enumerate(clients):
        ids = np.asarray(ids, dtype=np.int64)
        rng.shuffle(ids)
        if c < 10:
            a, b = int(len(ids) * .6), int(len(ids) * .8)
            split = {"train": ids[:a], "dev": ids[a:b], "audit": ids[b:]}
        else:
            a = int(len(ids) * .6)
            split = {"support": ids[:a], "query": ids[a:]}
        splits.append(split)
    return torch.from_numpy(x), torch.from_numpy(y), splits


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class Hyper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Emits all 2,410 parameters (weight and bias for each MLP layer).
        self.net = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(), torch.nn.Linear(64, 2410))

    def generate(self, z):
        p = self.net(z)
        w1 = p[:2048].reshape(32, 64)
        b1 = p[2048:2080]
        w2 = p[2080:2400].reshape(10, 32)
        b2 = p[2400:]
        return w1, b1, w2, b2


class System(torch.nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        if method == "pfedhn":
            self.hyper = Hyper()
            self.codes = torch.nn.Parameter(torch.zeros(12, 8))
        elif method in ("mirror", "film"):
            self.base = MLP()
            self.codes = torch.nn.Parameter(torch.zeros(12, 4))
            if method == "film":
                self.gamma_basis = torch.nn.Parameter(torch.zeros(4, 32))
                self.beta_basis = torch.nn.Parameter(torch.zeros(4, 32))
        elif method == "global":
            self.base = MLP()
        elif method == "independent":
            self.models = torch.nn.ModuleList([MLP() for _ in range(10)])
        else:
            raise ValueError(method)

    def logits(self, x, client=None, code=None):
        m = self.method
        if m == "pfedhn":
            z = code if code is not None else self.codes[client]
            w1, b1, w2, b2 = self.hyper.generate(z)
            return torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(x, w1, b1)), w2, b2)
        if m in ("mirror", "film"):
            h = torch.relu(self.base.fc1(x))
            if m == "mirror":
                theta = code if code is not None else self.codes[client]
                for j in range(4):
                    a, b = 2*j, 2*j+1
                    co, si = torch.cos(theta[j]), torch.sin(theta[j])
                    ha, hb = h[..., a], h[..., b]
                    h = h.clone()
                    h[..., a] = co*ha - si*hb
                    h[..., b] = si*ha + co*hb
            else:
                z = code if code is not None else self.codes[client]
                gamma = z @ self.gamma_basis
                beta = z @ self.beta_basis
                h = h * (1 + gamma) + beta
            return self.base.fc2(h)
        if m == "global":
            return self.base(x)
        return self.models[client](x)

    def local_code(self, client):
        if self.method == "pfedhn":
            return self.codes[client].detach().clone().requires_grad_(True)
        if self.method in ("mirror", "film"):
            return self.codes[client].detach().clone().requires_grad_(True)
        return None


def batches(indices, x, y, batch_size=128):
    for start in range(0, len(indices), batch_size):
        ii = torch.as_tensor(indices[start:start+batch_size], dtype=torch.long)
        yield x[ii], y[ii]


def loss_for(system, x, y, client, code=None):
    return torch.nn.functional.cross_entropy(system.logits(x, client, code), y)


def evaluate(system, x, y, ids, client, code=None):
    if len(ids) == 0:
        return float("nan"), float("nan")
    with torch.no_grad():
        xx = x[torch.as_tensor(ids, dtype=torch.long)]
        yy = y[torch.as_tensor(ids, dtype=torch.long)]
        p = torch.softmax(system.logits(xx, client, code), dim=-1).cpu().numpy()
    return float(accuracy_score(yy.numpy(), p.argmax(-1))), float(log_loss(yy.numpy(), p, labels=list(range(10))))


def train(system, x, y, splits, max_epochs=80):
    optimizer = torch.optim.AdamW(system.parameters(), lr=.003, weight_decay=1e-4)
    best, best_state, stale, updates = math.inf, None, 0, 0
    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        system.train()
        for client in range(10):
            idx = splits[client]["train"].copy()
            np.random.shuffle(idx)
            for xb, yb in batches(idx, x, y):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_for(system, xb, yb, client)
                loss.backward()
                optimizer.step()
                updates += 1
        system.eval()
        dev_losses = []
        for client in range(10):
            acc, ce = evaluate(system, x, y, splits[client]["dev"], client)
            if math.isfinite(ce):
                dev_losses.append(ce)
        score = float(np.mean(dev_losses))
        if score < best - 1e-8:
            best, stale = score, 0
            best_state = {k: v.detach().clone() for k, v in system.state_dict().items()}
        else:
            stale += 1
            if stale >= 8:
                break
    if best_state is not None:
        system.load_state_dict(best_state)
    return epoch + 1, updates, time.perf_counter() - t0


def adapt(system, x, y, support, client):
    """Update only unseen code (or independent full model) and snapshot 0/5/20."""
    method = system.method
    code = system.local_code(client)
    if method == "global":
        return {0: (None, 0.0)}, 0.0
    if method == "independent":
        model = MLP()
        # There is no trained row for an unseen client; initialize a fresh local model.
        params = list(model.parameters())
        call = lambda xx: model(xx)
        # Keep this branch independent; local model starts at seeded initialization.
        optimizer = torch.optim.Adam(params, lr=.01)
    else:
        params = [code]
        optimizer = torch.optim.Adam(params, lr=.01)
        call = lambda xx: system.logits(xx, client, code)
    xx = x[torch.as_tensor(support, dtype=torch.long)]
    yy = y[torch.as_tensor(support, dtype=torch.long)]
    snapshots = {0: ({k: v.detach().clone() for k, v in model.state_dict().items()} if method == "independent" else code.detach().clone(), 0.0)}
    elapsed = 0.0
    for step in range(1, 21):
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = call(xx)
        loss = torch.nn.functional.cross_entropy(logits, yy)
        loss.backward()
        optimizer.step()
        elapsed += time.perf_counter() - t0
        if step in (5, 20):
            snap = ({k: v.detach().clone() for k, v in model.state_dict().items()} if method == "independent" else code.detach().clone())
            snapshots[step] = (snap, elapsed)
    return snapshots, elapsed


def predict_at(system, x, ids, client, snapshot):
    if system.method == "global":
        code = None
    elif system.method == "independent":
        model = MLP()
        model.load_state_dict(snapshot)
        with torch.no_grad():
            xx = x[torch.as_tensor(ids, dtype=torch.long)]
            return model(xx)
    else:
        code = snapshot
    with torch.no_grad():
        return system.logits(x[torch.as_tensor(ids, dtype=torch.long)], client, code)


def metrics_from_snapshot(system, x, y, ids, client, snapshot):
    if len(ids) == 0:
        return float("nan"), float("nan")
    logits = predict_at(system, x, ids, client, snapshot)
    with torch.no_grad():
        p = torch.softmax(logits, dim=-1).cpu().numpy()
    yy = y[torch.as_tensor(ids, dtype=torch.long)].numpy()
    return float(accuracy_score(yy, p.argmax(-1))), float(log_loss(yy, p, labels=list(range(10))))


def tensor_payload_bytes(tensors, filename):
    # save_file emits exact inference payload serialization, including header/metadata.
    path = OUT / filename
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(path))
    return path.stat().st_size


def state_payload(system, world, model_seed):
    sd = system.state_dict()
    if system.method == "pfedhn":
        server = {k: v for k, v in sd.items() if k != "codes"}
        clients = {f"client_{i}_code": sd["codes"][i] for i in range(10)}
        full_model = 2410 * 4
        generation_macs = 8*64 + 64*2410
    elif system.method in ("mirror", "film"):
        server = {k: v for k, v in sd.items() if k != "codes"}
        clients = {f"client_{i}_code": sd["codes"][i] for i in range(10)}
        full_model = 2410 * 4
        generation_macs = 0
    elif system.method == "global":
        server, clients = dict(sd), {}
        full_model = 2410 * 4
        generation_macs = 0
    else:
        server = {}
        clients = {f"client_{i}.{k}": v for i,m in enumerate(system.models) for k,v in m.state_dict().items()}
        full_model = 2410 * 4
        generation_macs = 0
    server_bytes = tensor_payload_bytes(server, f"w{world}_s{model_seed}_{system.method}_server.safetensors") if server else 0
    client_bytes = tensor_payload_bytes(clients, f"w{world}_s{model_seed}_{system.method}_clients.safetensors") if clients else 0
    # Client provisioning payload is serialized alone, including key/format overhead.
    one_client_bytes = tensor_payload_bytes({"code": next(iter(clients.values()))}, f"w{world}_s{model_seed}_{system.method}_oneclient.safetensors") if system.method in ("pfedhn", "mirror", "film") else (full_model if system.method == "independent" else 0)
    return server_bytes, client_bytes, one_client_bytes, full_model, generation_macs


def run(world_seeds=None, model_seeds=None, max_epochs=80):
    OUT.mkdir(parents=True, exist_ok=True)
    world_seeds = world_seeds or [23,47,71]
    model_seeds = model_seeds or [31,47,59]
    rows = []
    for wi, ws in enumerate(world_seeds):
        x, y, splits = partition(ws)
        all_indices = np.concatenate([v for s in splits for v in s.values()])
        if len(np.unique(all_indices)) != len(y) or len(all_indices) != len(y):
            raise RuntimeError("partition is not a disjoint cover")
        np.savez_compressed(OUT / f"world_{ws}_splits.npz", **{f"c{c}_{k}":v for c,s in enumerate(splits) for k,v in s.items()})
        for mi, ms in enumerate(model_seeds):
            for method in METHODS:
                seed_all(ms)
                system = System(method)
                epochs, updates, train_wall = (0, 0, 0.0)
                if method != "independent":
                    epochs, updates, train_wall = train(system, x, y, splits, max_epochs)
                # Independent upper control: fit each local model only on that client's train split.
                else:
                    t0 = time.perf_counter()
                    for c in range(10):
                        opt = torch.optim.AdamW(system.models[c].parameters(), lr=.003, weight_decay=1e-4)
                        for ep in range(min(max_epochs, 80)):
                            for xb,yb in batches(splits[c]["train"], x, y):
                                opt.zero_grad(set_to_none=True); loss=loss_for(system,xb,yb,c); loss.backward(); opt.step(); updates += 1
                    epochs, train_wall = min(max_epochs,80), time.perf_counter()-t0
                server_b, clients_b, per_client_b, full_b, gen_macs = state_payload(system, ws, ms)
                # Evaluate seen-client audit and client adaptation/query.
                for c in range(10):
                    a, ce = evaluate(system, x, y, splits[c]["audit"], c)
                    rows.append(dict(world_seed=ws,model_seed=ms,method=method,phase="seen_audit",client=c,step=0,epochs=epochs,updates=updates,train_examples=sum(len(splits[k]["train"]) for k in range(10)),support_examples=0,query_examples=len(splits[c]["audit"]),server_payload_bytes=server_b,registered_client_payload_bytes=clients_b,per_client_download_bytes=per_client_b,generated_full_model_bytes=full_b,heldout_clients=2,accuracy=a,cross_entropy=ce,client_generation_macs=gen_macs,cpu_batch1_p95_ms=float("nan"),train_wall_seconds=train_wall,adapt_wall_seconds=0.0))
                for c in (10,11):
                    snaps, adapt_wall = adapt(system,x,y,splits[c]["support"],c)
                    for step,(snap,elapsed) in snaps.items():
                        a,ce=metrics_from_snapshot(system,x,y,splits[c]["query"],c,snap)
                        rows.append(dict(world_seed=ws,model_seed=ms,method=method,phase="unseen_query",client=c,step=step,epochs=epochs,updates=updates,train_examples=sum(len(splits[k]["train"]) for k in range(10)),support_examples=len(splits[c]["support"]),query_examples=len(splits[c]["query"]),server_payload_bytes=server_b,registered_client_payload_bytes=clients_b,per_client_download_bytes=per_client_b,generated_full_model_bytes=full_b,heldout_clients=2,accuracy=a,cross_entropy=ce,client_generation_macs=gen_macs,cpu_batch1_p95_ms=float("nan"),train_wall_seconds=train_wall,adapt_wall_seconds=elapsed))
    path = OUT / "metrics.csv"
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return path


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--world-seeds",nargs="+",type=int)
    parser.add_argument("--model-seeds",nargs="+",type=int)
    parser.add_argument("--max-epochs",type=int,default=80)
    args=parser.parse_args()
    print(run(args.world_seeds,args.model_seeds,args.max_epochs))
