#!/usr/bin/env python3
"""Controlled sparse interpolation coefficient-composition screen (MA-790)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
PROTOCOL = json.loads((ROOT / "PROTOCOL.json").read_text())
K, D, O, TOPK, RANK = 12, 16, 8, 3, 2
TRAIN_PAIRS = [(m, s) for m in range(4) for s in range(4) if (m + s) % 2 == 0]
AUDIT_PAIRS = [(m, s) for m in range(4) for s in range(4) if (m + s) % 2 == 1]
METHODS = ("native_simoe", "mirror_factorized", "ordinary_low_rank", "no_code")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)


def make_world(world_seed: int, coefficient_world: str):
    rng = np.random.default_rng(world_seed)
    anchors = rng.normal(0, 1 / math.sqrt(D), size=(K, O, D)).astype(np.float32)
    biases = rng.normal(0, .05, size=(K, O)).astype(np.float32)
    if coefficient_world == "aligned":
        ad = rng.normal(0, .65, size=(K, RANK))
        ass = rng.normal(0, .65, size=(K, RANK))
        zd = rng.normal(0, 1, size=(4, RANK))
        zs = rng.normal(0, 1, size=(4, RANK))
        base = rng.normal(0, .15, size=K)
        logits = np.stack([base + ad @ zd[m] + ass @ zs[s] for m in range(4) for s in range(4)])
    elif coefficient_world == "offorbit":
        logits = rng.normal(0, .8, size=(16, K))
    else:
        raise ValueError(coefficient_world)
    coeff = np.stack([topk_coeff_numpy(row) for row in logits]).astype(np.float32)
    return anchors, biases, coeff


def topk_coeff_numpy(logits: np.ndarray) -> np.ndarray:
    selected = np.argpartition(logits, -TOPK)[-TOPK:]
    vals = logits[selected] / .5
    p = np.exp(vals - vals.max()); p /= p.sum()
    out = np.zeros(K, dtype=np.float32); out[selected] = p
    return out


def sparse_topk_softmax(logits: torch.Tensor, straight_through: bool = False) -> torch.Tensor:
    soft = torch.softmax(logits / .5, dim=-1)
    idx = torch.topk(logits, TOPK, dim=-1).indices
    mask = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
    hard = soft * mask
    hard = hard / hard.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    if straight_through:
        return hard.detach() - soft.detach() + soft
    return hard


def pair_task_index(m: int, s: int) -> int:
    return m * 4 + s


def anchor_outputs(x: np.ndarray, anchors: np.ndarray, biases: np.ndarray) -> np.ndarray:
    # [N,K,O] outputs from all fixed shared physical anchors.
    return np.einsum("koi,ni->nko", anchors, x, optimize=True) + biases[None, :, :]


def task_samples(world_seed: int, task_idx: int, split: str, count: int, coeff: np.ndarray, anchors, biases):
    split_code = {"train": 101, "dev": 211, "support": 307, "query": 401}[split]
    rng = np.random.default_rng(world_seed * 1_000_003 + task_idx * 10_007 + split_code)
    x = rng.normal(0, 1, size=(count, D)).astype(np.float32)
    f = anchor_outputs(x, anchors, biases)
    y = np.einsum("k,nko->no", coeff, f, optimize=True).astype(np.float32)
    return x, y


class CoefficientModel(torch.nn.Module):
    def __init__(self, method: str):
        super().__init__(); self.method = method
        if method == "native_simoe":
            self.rows = torch.nn.Parameter(torch.zeros(16, K))
        elif method == "mirror_factorized":
            # Polar coordinates retain two degrees of freedom per factor ID.
            self.domain_radius_raw = torch.nn.Parameter(torch.full((4, 1), .54))
            self.domain_angle = torch.nn.Parameter(torch.randn(4, 1) * .1)
            self.skill_radius_raw = torch.nn.Parameter(torch.full((4, 1), .54))
            self.skill_angle = torch.nn.Parameter(torch.randn(4, 1) * .1)
            self.domain_basis = torch.nn.Parameter(torch.randn(K, RANK) * .1)
            self.skill_basis = torch.nn.Parameter(torch.randn(K, RANK) * .1)
            self.bias = torch.nn.Parameter(torch.zeros(K))
        elif method == "ordinary_low_rank":
            self.domain_codes = torch.nn.Parameter(torch.randn(4, RANK) * .1)
            self.skill_codes = torch.nn.Parameter(torch.randn(4, RANK) * .1)
            self.domain_basis = torch.nn.Parameter(torch.randn(K, RANK) * .1)
            self.skill_basis = torch.nn.Parameter(torch.randn(K, RANK) * .1)
            self.bias = torch.nn.Parameter(torch.zeros(K))
        elif method == "no_code":
            self.logits = torch.nn.Parameter(torch.zeros(K))
        else: raise ValueError(method)

    def coefficient_logits(self, m: int, s: int):
        if self.method == "native_simoe": return self.rows[pair_task_index(m, s)]
        if self.method == "no_code": return self.logits
        if self.method == "mirror_factorized":
            rd = torch.nn.functional.softplus(self.domain_radius_raw[m])
            rs = torch.nn.functional.softplus(self.skill_radius_raw[s])
            zd = rd * torch.cat((torch.cos(self.domain_angle[m]), torch.sin(self.domain_angle[m])))
            zs = rs * torch.cat((torch.cos(self.skill_angle[s]), torch.sin(self.skill_angle[s])))
        else:
            zd, zs = self.domain_codes[m], self.skill_codes[s]
        return self.bias + self.domain_basis @ zd + self.skill_basis @ zs

    def coefficients(self, m: int, s: int, straight_through=False):
        return sparse_topk_softmax(self.coefficient_logits(m, s), straight_through)


def predict(model, method, m, s, x, anchors, biases):
    coeff = model.coefficients(m, s).detach().cpu().numpy()
    f = anchor_outputs(x, anchors, biases)
    return np.einsum("k,nko->no", coeff, f, optimize=True), coeff


def mse(a, b): return float(np.mean(np.square(a - b)))


def fit_seen(model: CoefficientModel, coefficient_world: str, world_seed: int, anchors, biases, target_coeffs, max_updates=2000):
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    rng = np.random.default_rng(world_seed * 811 + 1201)
    cache = {}
    for m, s in TRAIN_PAIRS:
        x, y = task_samples(world_seed, pair_task_index(m, s), "train", 2048, target_coeffs[pair_task_index(m, s)], anchors, biases)
        cache[(m,s)] = (anchor_outputs(x, anchors, biases), y)
    dev_cache = {}
    for m, s in TRAIN_PAIRS:
        x, y = task_samples(world_seed, pair_task_index(m, s), "dev", 512, target_coeffs[pair_task_index(m, s)], anchors, biases)
        dev_cache[(m,s)] = (anchor_outputs(x, anchors, biases), y)
    best, best_state, stale, updates = float("inf"), None, 0, 0
    start = time.perf_counter()
    while updates < max_updates:
        m, s = TRAIN_PAIRS[int(rng.integers(0, len(TRAIN_PAIRS)))]
        fout, y = cache[(m,s)]
        ix = rng.integers(0, len(y), size=256)
        fbatch = torch.as_tensor(fout[ix], dtype=torch.float32)
        ybatch = torch.as_tensor(y[ix], dtype=torch.float32)
        c = model.coefficients(m, s, straight_through=True)
        pred = torch.einsum("k,nko->no", c, fbatch)
        loss = torch.nn.functional.mse_loss(pred, ybatch)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); updates += 1
        if updates % 25 == 0:
            with torch.no_grad():
                dev_losses=[]
                for (dm,ds),(df,dy) in dev_cache.items():
                    cdev=model.coefficients(dm,ds)
                    preddev=torch.einsum("k,nko->no",cdev,torch.as_tensor(df,dtype=torch.float32))
                    dev_losses.append(torch.nn.functional.mse_loss(preddev,torch.as_tensor(dy)).item())
                score=float(np.mean(dev_losses))
            if score < best - 1e-9:
                best=score; stale=0; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= 200: break
    if best_state is not None: model.load_state_dict(best_state)
    return updates, sum(len(y) for _,y in cache.values()), best, time.perf_counter()-start


def adapt_native(model, m, s, anchors, biases, target_coeff, world_seed, checkpoints=(0,25,100,300)):
    support_x, support_y = task_samples(world_seed, pair_task_index(m,s), "support", 64, target_coeff, anchors, biases)
    query_x, query_y = task_samples(world_seed, pair_task_index(m,s), "query", 2048, target_coeff, anchors, biases)
    local = torch.nn.Parameter(torch.zeros(K))
    optimizer = torch.optim.Adam([local], lr=.03)
    cache = anchor_outputs(support_x, anchors, biases)
    qcache = anchor_outputs(query_x, anchors, biases)
    out=[]
    def evaluate(step):
        c=sparse_topk_softmax(local.detach()).numpy()
        pred=np.einsum("k,nko->no",c,qcache,optimize=True)
        return {"stage":f"support_{step}","updates":step,"query_mse":mse(pred,query_y),"coeff_reconstruction_mse":mse(c,target_coeff)}
    out.append(evaluate(0))
    for step in range(1,301):
        c=sparse_topk_softmax(local,straight_through=True)
        pred=torch.einsum("k,nko->no",c,torch.as_tensor(cache,dtype=torch.float32))
        loss=torch.nn.functional.mse_loss(pred,torch.as_tensor(support_y,dtype=torch.float32))
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step in checkpoints[1:]: out.append(evaluate(step))
    return out, local.detach()


def serialize_model(model, method, anchors, biases, coefficient_world, world, init_seed):
    ART.mkdir(parents=True,exist_ok=True)
    stem=f"{coefficient_world}_w{world}_s{init_seed}_{method}"
    def packed(tensors):
        items=sorted(tensors.items())
        flat=torch.cat([v.reshape(-1) for _,v in items]).to(torch.float32)
        layout={k:list(v.shape) for k,v in items}
        return flat,layout
    anchor_path=ART/f"{stem}_anchors.safetensors"
    anchor_flat,anchor_layout=packed({"weights":torch.as_tensor(anchors),"biases":torch.as_tensor(biases)})
    save_file({"p":anchor_flat},str(anchor_path),metadata={"schema":"MA790-PAYLOAD-V1","group":"anchors"})
    sd={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    if method=="mirror_factorized":
        code_keys={k:v for k,v in sd.items() if "radius" in k or "angle" in k}
        gen={k:v for k,v in sd.items() if k not in code_keys}
    elif method=="ordinary_low_rank":
        code_keys={k:v for k,v in sd.items() if "codes" in k}
        gen={k:v for k,v in sd.items() if k not in code_keys}
    else:
        code_keys={}
        gen=sd
    gen_path=ART/f"{stem}_generator.safetensors"
    gen_flat,gen_layout=packed(gen)
    save_file({"p":gen_flat},str(gen_path),metadata={"schema":"MA790-PAYLOAD-V1","group":"generator"})
    code_bytes=0
    if code_keys:
        cp=ART/f"{stem}_factor_codes.safetensors";code_flat,code_layout=packed(code_keys);save_file({"p":code_flat},str(cp),metadata={"schema":"MA790-PAYLOAD-V1","group":"factor_codes"});code_bytes=cp.stat().st_size
    else:
        code_layout={}
    meta={"method":method,"coefficient_world":coefficient_world,"task_grid":"4x4","top_k":3,"anchor_layout":anchor_layout,"generator_layout":gen_layout,"factor_code_layout":code_layout}
    mp=ART/f"{stem}_metadata.json";mp.write_text(json.dumps(meta,sort_keys=True,separators=(",",":"))+"\n")
    anchor_bytes=anchor_path.stat().st_size;generator_bytes=gen_path.stat().st_size+mp.stat().st_size
    full16=ART/f"{stem}_free16_coefficient_reference.safetensors"
    save_file({"p":torch.zeros(16*K)},str(full16),metadata={"schema":"MA790-PAYLOAD-V1","group":"free16_coefficients"})
    return anchor_bytes,generator_bytes,code_bytes,anchor_bytes+generator_bytes+code_bytes,full16.stat().st_size


def coefficient_macs(method):
    if method=="native_simoe": return 0
    if method=="no_code": return 0
    return 2*K*RANK + 2*RANK


def inference_latency(model, method):
    for _ in range(30): model.coefficients(0,0)
    start=time.perf_counter()
    for _ in range(500): model.coefficients(0,0)
    return 1000*(time.perf_counter()-start)/500


def run(world_seeds=None, model_seeds=None, max_updates=2000, audit=False):
    ART.mkdir(parents=True,exist_ok=True)
    world_seeds=world_seeds or [11,23,37];model_seeds=model_seeds or [31,47,59]
    rows=[]; data_hashes={}
    for world in world_seeds:
        for cworld in ("aligned","offorbit"):
            anchors,anchor_bias,target_coeffs=make_world(world,cworld)
            # Save only task definitions and fixed shared anchors; query examples are generated after training.
            for init_seed in model_seeds:
                trained={}; train_meta={}
                for method in METHODS:
                    set_seed(init_seed)
                    model=CoefficientModel(method)
                    updates,ntrain,dev_mse,train_wall=fit_seen(model,cworld,world,anchors,anchor_bias,target_coeffs,max_updates)
                    trained[method]=model
                    train_meta[method]=(updates,ntrain,dev_mse,train_wall)
                # Checkpoints are now frozen from development tasks. Audit query samples are first created below.
                eval_pairs=AUDIT_PAIRS if audit else TRAIN_PAIRS
                for method in METHODS:
                    model=trained[method]
                    updates,ntrain,dev_mse,train_wall=train_meta[method]
                    anchor_b,gen_b,code_b,total_b,free16_b=serialize_model(model,method,anchors,anchor_bias,cworld,world,init_seed)
                    latency=inference_latency(model,method)
                    if audit:
                        for m,s in AUDIT_PAIRS:
                            ti=pair_task_index(m,s);target=target_coeffs[ti]
                            # The independent SIMoE control adapts a free sparse coefficient row on support.
                            if method=="native_simoe":
                                stages,local=adapt_native(model,m,s,anchors,anchor_bias,target,world)
                                with torch.no_grad():
                                    model.rows[ti].copy_(local)
                            elif method=="mirror_factorized":
                                stages=[{"stage":"zero_shot","updates":0}]
                                qx,qy=task_samples(world,ti,"query",2048,target,anchors,anchor_bias)
                                pred,c=predict(model,method,m,s,qx,anchors,anchor_bias)
                                stages[0].update(query_mse=mse(pred,qy),coeff_reconstruction_mse=mse(c,target))
                            elif method=="ordinary_low_rank":
                                stages=[{"stage":"zero_shot","updates":0}]
                                qx,qy=task_samples(world,ti,"query",2048,target,anchors,anchor_bias)
                                pred,c=predict(model,method,m,s,qx,anchors,anchor_bias)
                                stages[0].update(query_mse=mse(pred,qy),coeff_reconstruction_mse=mse(c,target))
                            else:
                                stages=[{"stage":"zero_shot","updates":0}]
                                qx,qy=task_samples(world,ti,"query",2048,target,anchors,anchor_bias)
                                pred,c=predict(model,method,m,s,qx,anchors,anchor_bias)
                                stages[0].update(query_mse=mse(pred,qy),coeff_reconstruction_mse=mse(c,target))
                            for stage in stages:
                                rows.append({"world_seed":world,"model_seed":init_seed,"coefficient_world":cworld,"method":method,"task_pair":f"d{m}_s{s}","stage":stage["stage"],"updates":stage["updates"],"examples_seen":256*updates + (64*stage["updates"] if method=="native_simoe" else 0),"query_mse":stage["query_mse"],"coeff_reconstruction_mse":stage["coeff_reconstruction_mse"],"anchor_payload_bytes":anchor_b,"shared_generator_payload_bytes":gen_b,"registered_code_payload_bytes":code_b,"total_inference_payload_bytes":total_b,"free_16row_coeff_payload_bytes":free16_b,"coefficient_state_ratio":(gen_b+code_b)/free16_b,"coeff_generation_macs":coefficient_macs(method),"cpu_batch1_latency_ms":latency,"train_wall_seconds":train_wall,"status":"audit"})
                    else:
                        for m,s in TRAIN_PAIRS:
                            x,y=task_samples(world,pair_task_index(m,s),"dev",512,target_coeffs[pair_task_index(m,s)],anchors,anchor_bias)
                            pred,c=predict(model,method,m,s,x,anchors,anchor_bias)
                            rows.append({"world_seed":world,"model_seed":init_seed,"coefficient_world":cworld,"method":method,"task_pair":f"d{m}_s{s}","stage":"development","updates":updates,"examples_seen":256*updates,"query_mse":mse(pred,y),"coeff_reconstruction_mse":mse(c,target_coeffs[pair_task_index(m,s)]),"anchor_payload_bytes":anchor_b,"shared_generator_payload_bytes":gen_b,"registered_code_payload_bytes":code_b,"total_inference_payload_bytes":total_b,"free_16row_coeff_payload_bytes":free16_b,"coefficient_state_ratio":(gen_b+code_b)/free16_b,"coeff_generation_macs":coefficient_macs(method),"cpu_batch1_latency_ms":latency,"train_wall_seconds":train_wall,"status":"development_only"})
                    if audit and method == "native_simoe":
                        # Persist the actual support-adapted held-out coefficient rows.
                        serialize_model(model, method, anchors, anchor_bias, cworld, world, init_seed)
    out=ART/("metrics_audit.csv" if audit else "metrics_development.csv")
    with out.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return out


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--world-seeds",nargs="+",type=int);ap.add_argument("--model-seeds",nargs="+",type=int);ap.add_argument("--max-updates",type=int,default=2000);ap.add_argument("--audit",action="store_true");a=ap.parse_args()
    print(run(a.world_seeds,a.model_seeds,a.max_updates,a.audit))
