"""Synthetic R-GCN relation-transform mechanism screen for MA-742."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "source"
A_VALUES = B_VALUES = 4
N_REL = A_VALUES * B_VALUES
N_HOLDOUT = {0, 6, 11, 13}  # (a,b): (0,0),(1,2),(2,3),(3,1)
D_IN = D_OUT = 8
N_BASIS = 4
TRAIN_PER_REL = 256
EVAL_PER_REL = 256
UPDATES = int(os.environ.get("MA742_UPDATES", "150"))
LR = 0.025


def relation_attrs():
    return [(r // B_VALUES, r % B_VALUES) for r in range(N_REL)]


def split_relations():
    train = [r for r in range(N_REL) if r not in N_HOLDOUT]
    return train, sorted(N_HOLDOUT)


def make_world(seed: int):
    g = torch.Generator().manual_seed(seed)
    true_basis = torch.randn(N_BASIS, D_IN, D_OUT, generator=g) / math.sqrt(D_IN * N_BASIS)
    u = torch.randn(A_VALUES, 2, generator=g) * 0.8
    v = torch.randn(B_VALUES, 2, generator=g) * 0.8
    decoder = torch.randn(2, N_BASIS, generator=g) * 0.65
    attrs = relation_attrs()
    codes = torch.stack([(u[a] * v[b]) @ decoder for a, b in attrs])
    weights = torch.einsum("rk,kio->rio", codes, true_basis)
    return weights, true_basis


def make_data(seed: int, weights: torch.Tensor, n_per_relation: int):
    g = torch.Generator().manual_seed(seed)
    xs, ys, rs = [], [], []
    for r in range(N_REL):
        x = torch.randn(n_per_relation, D_IN, generator=g)
        noise = 0.015 * torch.randn(n_per_relation, D_OUT, generator=g)
        y = x @ weights[r] + noise
        xs.append(x)
        ys.append(y)
        rs.extend([r] * n_per_relation)
    return torch.cat(xs), torch.cat(ys), torch.tensor(rs, dtype=torch.long)


class RelationModel(nn.Module):
    def __init__(self, method: str, rank: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.method, self.rank = method, rank
        if method == "independent":
            self.weights = nn.Parameter(torch.randn(N_REL, D_IN, D_OUT) * 0.04)
            self.register_parameter("basis", None)
        else:
            self.basis = nn.Parameter(torch.randn(N_BASIS, D_IN, D_OUT) * 0.04)
        if method == "native":
            self.coefficients = nn.Parameter(torch.randn(N_REL, N_BASIS) * 0.04)
        elif method in ("additive", "mirror"):
            self.attr_a = nn.Parameter(torch.randn(A_VALUES, rank) * 0.04)
            self.attr_b = nn.Parameter(torch.randn(B_VALUES, rank) * 0.04)
            self.decode = nn.Parameter(torch.randn(rank, N_BASIS) * 0.04)

    def relation_coeff(self, rels):
        if self.method == "native":
            return self.coefficients[rels]
        if self.method == "independent":
            return None
        attrs = relation_attrs()
        ab = torch.tensor(attrs, device=rels.device)
        a, b = ab[rels, 0], ab[rels, 1]
        if self.method == "additive":
            code = self.attr_a[a] + self.attr_b[b]
        else:
            code = self.attr_a[a] * self.attr_b[b]
        return code @ self.decode

    def forward(self, x, rels):
        if self.method == "independent":
            w = self.weights[rels]
            return torch.bmm(x.unsqueeze(1), w).squeeze(1)
        coeff = self.relation_coeff(rels)
        transformed = torch.einsum("bi,kio->bko", x, self.basis)
        return torch.einsum("bk,bko->bo", coeff, transformed)


def train_one(method: str, rank: int, seed: int, train_seed: int):
    weights, _ = make_world(seed)
    x, y, rels = make_data(train_seed, weights, TRAIN_PER_REL)
    train_rels, _ = split_relations()
    keep = torch.zeros(N_REL, dtype=torch.bool)
    keep[train_rels] = True
    mask = keep[rels]
    x, y, rels = x[mask], y[mask], rels[mask]
    model = RelationModel(method, rank, seed + 10_000 + train_seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    start = time.perf_counter()
    for _ in range(UPDATES):
        opt.zero_grad(set_to_none=True)
        loss = (model(x, rels) - y).square().mean()
        loss.backward()
        opt.step()
    wall = time.perf_counter() - start
    return model, wall, len(x) * UPDATES, float(loss.detach())


def evaluate(model, world_seed: int, data_seed: int):
    weights, _ = make_world(world_seed)
    x, y, rels = make_data(data_seed, weights, EVAL_PER_REL)
    train_rel, held_rel = split_relations()
    held_mask = torch.zeros_like(rels, dtype=torch.bool)
    seen_mask = torch.zeros_like(rels, dtype=torch.bool)
    held_mask[torch.isin(rels, torch.tensor(held_rel))] = True
    seen_mask[torch.isin(rels, torch.tensor(train_rel))] = True
    model.eval()
    with torch.inference_mode():
        t0 = time.perf_counter()
        pred = model(x, rels)
        elapsed = time.perf_counter() - t0
    errs = (pred - y).square().mean(dim=1)
    return float(errs[held_mask].mean()), float(errs[seen_mask].mean()), elapsed


def state_payload(model, method: str):
    sd = model.state_dict()
    if method == "independent":
        marginal = {"weights": sd["weights"]}
        common = {}
    else:
        common = {"basis": sd["basis"]}
        marginal = {k: v for k, v in sd.items() if k != "basis"}
    metadata = {"method": method, "rank": model.rank, "n_relations": N_REL,
                "input_dim": D_IN, "output_dim": D_OUT, "n_basis": N_BASIS,
                "heldout_relation_ids": sorted(N_HOLDOUT), "format": "MA-742-v1"}
    def dump(obj):
        bio = io.BytesIO()
        torch.save(obj, bio)
        return bio.getvalue()
    whole_bytes = dump({"metadata": metadata, "state_dict": sd})
    marginal_bytes = dump({"metadata": metadata, "coordinate_state": marginal})
    common_bytes = dump({"metadata": metadata, "shared_state": common}) if common else b""
    return whole_bytes, marginal_bytes, common_bytes, hashlib.sha256(whole_bytes).hexdigest()


def mac_proxy(method: str, rank: int, batch: int):
    if method == "independent":
        return batch * D_IN * D_OUT
    # MACs for input-to-basis transforms, coefficient composition and weighted sum.
    transform = batch * N_BASIS * D_IN * D_OUT
    compose = (A_VALUES + B_VALUES) * rank * N_BASIS + rank * N_BASIS
    apply = batch * N_BASIS * D_OUT
    return transform + compose + apply


def run_stage(stage: str, seeds: list[int], ranks: list[int]):
    rows, models_for_rank = [], {}
    methods = [("independent", 4), ("native", 4)]
    if stage == "development":
        for r in ranks:
            methods += [("additive", r), ("mirror", r)]
    else:
        frozen = json.loads((OUT / "frozen_config.json").read_text())
        r = int(frozen["selected_rank"])
        methods += [("additive", r), ("mirror", r)]
    normalized = methods
    for world_seed in seeds:
        train_seed = world_seed + 1
        for method, rank in normalized:
            model, train_wall, seen_exposures, final_loss = train_one(method, rank, world_seed, train_seed)
            # deterministic evaluation dataset seed is separate from training seed
            held_mse, seen_mse, infer_wall = evaluate(model, world_seed, world_seed + 2)
            whole, marginal, common, digest = state_payload(model, method)
            row = {'world_seed':world_seed,'method':method,'latent_rank':rank,'serialized_bytes':len(whole),
                   'marginal_coordinate_bytes':len(marginal),'shared_basis_bytes':len(common),
                   'payload_sha256':digest,'train_examples':seen_exposures,'optimizer_updates':UPDATES,
                   'train_macs_proxy':mac_proxy(method,rank,len(split_relations()[0])*TRAIN_PER_REL)*UPDATES*3,
                   'train_wall_s':train_wall,'inference_macs_proxy':mac_proxy(method,rank,EVAL_PER_REL*N_REL),
                   'inference_wall_s':infer_wall,'heldout_combo_mse':held_mse,'seen_relation_mse':seen_mse,
                   'final_train_mse':final_loss,'stage':stage}
            rows.append(row)
            if stage == 'development':
                models_for_rank[(world_seed,method,rank)] = row
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['development','fresh'],required=True)
    args=ap.parse_args()
    seeds = [7421,7422] if args.stage=='development' else [74201,74202,74203]
    ranks=[2,4,8]
    rows=run_stage(args.stage,seeds,ranks)
    out=OUT/f'{args.stage}_{UPDATES}_updates.json'
    out.write_text(json.dumps({'stage':args.stage,'seeds':seeds,'rows':rows},indent=2)+'\n')
    print(json.dumps({'stage':args.stage,'rows':len(rows),'output':str(out)},indent=2))

if __name__=='__main__': main()
