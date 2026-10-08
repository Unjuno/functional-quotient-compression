#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

from model import ARCHS, METHODS, OFASupernet, IndependentSubnet, macs_per_example, correction_macs_per_example

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
BATCH = 64
UPDATES = 500
CORRECTION_UPDATES = 100
FIELDS = ["condition", "world_or_seed", "learning_rate", "method", "architecture", "serialized_bytes", "train_example_views", "optimizer_updates", "active_MAC_proxy", "adapt_MAC_proxy", "train_wall_time_s", "inference_examples_per_second", "accuracy", "nll", "payload_sha256", "roundtrip_max_abs_diff", "status_note"]
DATA_HASH = "faf2d98f61250c1cdc0b114323133d3c58da04f5c5d28bb78e4bde3c936a75b1"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_world(seed: int) -> dict[str, torch.Tensor | int]:
    data = load_digits()
    digest = hashlib.sha256(data.data.astype("float32").tobytes() + data.target.astype("int64").tobytes()).hexdigest()
    if digest != DATA_HASH or sklearn.__version__ != "1.8.0":
        raise RuntimeError(f"dataset provenance mismatch: sklearn={sklearn.__version__}, sha256={digest}")
    ids = np.arange(len(data.target))
    train_ids, rest = train_test_split(ids, train_size=0.60, stratify=data.target, random_state=seed)
    val_ids, test_ids = train_test_split(rest, test_size=0.50, stratify=data.target[rest], random_state=seed + 1)
    x = torch.tensor(data.data.astype("float32") / 16.0)
    y = torch.tensor(data.target.astype("int64"))
    return {
        "x": x,
        "y": y,
        "train_ids": torch.tensor(train_ids, dtype=torch.long),
        "val_ids": torch.tensor(val_ids, dtype=torch.long),
        "test_ids": torch.tensor(test_ids, dtype=torch.long),
        "seed": seed,
    }


def split_tensors(world: dict, part: str) -> tuple[torch.Tensor, torch.Tensor]:
    ids = world[f"{part}_ids"]
    return world["x"][ids], world["y"][ids]


def draw_batch(world: dict, n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    ids = world["train_ids"]
    take = ids[torch.randint(len(ids), (n,), generator=gen)]
    return world["x"][take], world["y"][take]


def soft_distill(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return F.kl_div(F.log_softmax(student / 2.0, dim=-1), F.softmax(teacher.detach() / 2.0, dim=-1), reduction="batchmean") * 4.0


def train_supernet(world: dict, seed: int, lr: float) -> tuple[OFASupernet, float, int]:
    seed_all(seed)
    model = OFASupernet()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed + 101)
    random_arches = ARCHS[1:-1]
    t0 = time.perf_counter()
    model.train()
    for _ in range(UPDATES):
        xb, yb = draw_batch(world, BATCH, gen)
        teacher = model(xb, (64, 2))
        small = model(xb, (16, 1))
        arch = random_arches[int(torch.randint(len(random_arches), (), generator=gen))]
        sampled = model(xb, arch)
        loss = F.cross_entropy(teacher, yb) + F.cross_entropy(small, yb) + F.cross_entropy(sampled, yb)
        loss = loss + 0.25 * (soft_distill(small, teacher) + soft_distill(sampled, teacher))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    mac = UPDATES * BATCH * (macs_per_example((64, 2)) + macs_per_example((16, 1)) + sum(macs_per_example(a) for a in random_arches) / len(random_arches))
    return model, time.perf_counter() - t0, int(mac)


def fit_code(model: OFASupernet, world: dict, arch: tuple[int, int], method: str, seed: int) -> tuple[dict[str, torch.Tensor], float, int]:
    if method == "ofa_mirror":
        params = {"angles": torch.nn.Parameter(torch.zeros(4))}
    elif method == "ofa_film":
        params = {"scales": torch.nn.Parameter(torch.ones(4))}
    elif method == "ofa_lora":
        width, _ = arch
        a = torch.nn.Parameter(torch.randn(width, 1) * 0.01)
        b = torch.nn.Parameter(torch.zeros(1, 10))
        params = {"a": a, "b": b}
    else:
        raise ValueError(method)
    opt = torch.optim.Adam(params.values(), lr=0.01)
    gen = torch.Generator().manual_seed(seed + 1009 + arch[0] * 13 + arch[1])
    x_all, y_all = split_tensors(world, "train")
    t0 = time.perf_counter()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for _ in range(CORRECTION_UPDATES):
        ids = torch.randint(len(y_all), (BATCH,), generator=gen)
        xb, yb = x_all[ids], y_all[ids]
        h = model.hidden(xb, *arch)
        if method in ("ofa_mirror", "ofa_film"):
            code = {k: v for k, v in params.items()}
            from model import apply_code
            h2 = apply_code(h, method, code)
            logits = F.linear(h2, model.head[:, :arch[0]], model.head_bias)
        else:
            base = model(xb, arch)
            logits = base + (h @ params["a"]) @ params["b"]
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    duration = time.perf_counter() - t0
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return {k: v.detach().clone() for k, v in params.items()}, duration, CORRECTION_UPDATES * BATCH


def train_independent(world: dict, seed: int, lr: float) -> tuple[dict[tuple[int, int], IndependentSubnet], float, int]:
    models = {}
    total_time = 0.0
    total_mac = 0
    for ai, arch in enumerate(ARCHS):
        seed_all(seed + 5000 + ai * 37)
        model = IndependentSubnet(*arch)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        gen = torch.Generator().manual_seed(seed + 7000 + ai)
        model.train()
        t0 = time.perf_counter()
        for _ in range(UPDATES):
            xb, yb = draw_batch(world, BATCH, gen)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        total_time += time.perf_counter() - t0
        total_mac += UPDATES * BATCH * macs_per_example(arch)
        model.eval()
        models[arch] = model
    return models, total_time, total_mac


def make_bundle(method: str, shared: OFASupernet | None, codes: dict, independent: dict | None, metadata: dict) -> tuple[bytes, dict]:
    bundle = {
        "method": method,
        "metadata": metadata,
        "shared_state": None if shared is None else {k: v.detach().cpu() for k, v in shared.state_dict().items()},
        "codes": {str(k): {n: v.detach().cpu() for n, v in code.items()} for k, code in codes.items()},
        "independent_states": None if independent is None else {str(k): {n: v.detach().cpu() for n, v in m.state_dict().items()} for k, m in independent.items()},
    }
    b = io.BytesIO()
    torch.save(bundle, b)
    payload = b.getvalue()
    loaded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    # Validate deterministic inference parity for every registered subnet before scoring.
    shared_rt, codes_rt, indep_rt = runtime_from_bundle(loaded)
    check_x = torch.linspace(0, 1, 5 * 64).reshape(5, 64)
    roundtrip_max_diff = 0.0
    for arch in ARCHS:
        before = predict(shared, codes, independent or {}, method, arch, check_x)
        after = predict(shared_rt, codes_rt, indep_rt, method, arch, check_x)
        diff = float((before.detach() - after.detach()).abs().max())
        roundtrip_max_diff = max(roundtrip_max_diff, diff)
        if diff > 0.0:
            raise RuntimeError(f"non-exact inference payload roundtrip for {method} {arch}: {diff}")
    return payload, loaded, roundtrip_max_diff


def runtime_from_bundle(bundle: dict) -> tuple[OFASupernet | None, dict, dict]:
    method = bundle["method"]
    shared = None
    independent = {}
    codes = {tuple(map(int, k.strip("() ").split(","))): v for k, v in bundle["codes"].items()}
    if method != "independent":
        shared = OFASupernet()
        shared.load_state_dict(bundle["shared_state"])
        shared.eval()
    else:
        for key, state in bundle["independent_states"].items():
            arch = tuple(map(int, key.strip("() ").split(",")))
            net = IndependentSubnet(*arch)
            net.load_state_dict(state)
            net.eval()
            independent[arch] = net
    return shared, codes, independent


def predict(shared, codes, independent, method: str, arch: tuple[int, int], x: torch.Tensor) -> torch.Tensor:
    if method == "independent":
        return independent[arch](x)
    if method in ("ofa_mirror", "ofa_film"):
        return shared(x, arch, method, codes[arch])
    if method == "ofa_lora":
        h = shared.hidden(x, *arch)
        return F.linear(h, shared.head[:, :arch[0]], shared.head_bias) + (h @ codes[arch]["a"]) @ codes[arch]["b"]
    return shared(x, arch)


def evaluate_bundle(payload: bytes, world: dict, part: str, arch: tuple[int, int]) -> tuple[float, float, float, float]:
    bundle = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    shared, codes, independent = runtime_from_bundle(bundle)
    method = bundle["method"]
    x, y = split_tensors(world, part)
    with torch.no_grad():
        t0 = time.perf_counter()
        logits = predict(shared, codes, independent, method, arch, x[:min(256, len(y))])
        for _ in range(20):
            _ = predict(shared, codes, independent, method, arch, x[:min(256, len(y))])
        elapsed = time.perf_counter() - t0
        all_logits = predict(shared, codes, independent, method, arch, x)
        nll = float(F.cross_entropy(all_logits, y))
        acc = float((all_logits.argmax(-1) == y).float().mean())
    throughput = float(20 * min(256, len(y)) / max(elapsed, 1e-9))
    return acc, nll, throughput, float(logits.abs().max())


def run_world(seed: int, lr: float, condition: str) -> tuple[list[dict], dict]:
    world = make_world(seed)
    model_seed = seed + 20_000
    t0 = time.perf_counter()
    shared, shared_time, shared_mac = train_supernet(world, model_seed, lr)
    codes_by_method = {"ofa_mirror": {}, "ofa_film": {}, "ofa_lora": {}}
    code_time = {m: 0.0 for m in codes_by_method}
    code_views = {m: 0 for m in codes_by_method}
    code_mac = {m: 0 for m in codes_by_method}
    for method in codes_by_method:
        for arch in ARCHS:
            code, sec, views = fit_code(shared, world, arch, method, model_seed)
            codes_by_method[method][arch] = code
            code_time[method] += sec
            code_views[method] += views
            code_mac[method] += CORRECTION_UPDATES * BATCH * (macs_per_example(arch) + correction_macs_per_example(method, arch))
    independent, independent_time, independent_mac = train_independent(world, model_seed, lr)
    rows=[];payloads={};meta={"architectures":ARCHS,"dataset":"sklearn-digits","world_seed":seed,"normalization":"pixels/16","lr":lr}
    # Every method payload includes its entire deployable bank, and each architecture row charges that complete bank.
    for method in METHODS:
        if method == "ofa_shared":
            payload, bundle, roundtrip_diff = make_bundle(method, shared, {}, None, meta)
            train_time=shared_time;train_views=UPDATES*BATCH*3;updates=UPDATES;mac=shared_mac;adapt_mac=0
        elif method in codes_by_method:
            payload, bundle, roundtrip_diff = make_bundle(method, shared, codes_by_method[method], None, meta)
            train_time=shared_time+code_time[method];train_views=UPDATES*BATCH*3+code_views[method];updates=UPDATES+len(ARCHS)*CORRECTION_UPDATES;mac=shared_mac+code_mac[method];adapt_mac=code_mac[method]
        else:
            payload, bundle, roundtrip_diff = make_bundle(method, None, {}, independent, meta)
            train_time=independent_time;train_views=len(ARCHS)*UPDATES*BATCH;updates=len(ARCHS)*UPDATES;mac=independent_mac;adapt_mac=0
        payload_hash=hashlib.sha256(payload).hexdigest()
        path=OUT/'payloads'/condition/str(seed)/str(lr)/(method+'.pt')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
        for arch in ARCHS:
            acc,nll,tps,maxlog=evaluate_bundle(payload,world,"val" if condition=="development" else "test",arch)
            row={"condition":condition,"world_or_seed":seed,"learning_rate":lr,"method":method,"architecture":f"w{arch[0]}_d{arch[1]}","serialized_bytes":len(payload),"train_example_views":train_views,"optimizer_updates":updates,"active_MAC_proxy":mac,"adapt_MAC_proxy":adapt_mac,"train_wall_time_s":round(train_time,6),"inference_examples_per_second":round(tps,3),"accuracy":f"{acc:.8f}","nll":f"{nll:.10f}","payload_sha256":payload_hash,"roundtrip_max_abs_diff":roundtrip_diff,"status_note":f"digits world; lr={lr}; payload={path.name}; max_logit={maxlog:.6g}"}
            rows.append(row)
    return rows,{"world":world,"shared":shared,"shared_time":shared_time,"shared_mac":shared_mac,"codes":codes_by_method,"independent":independent,"payloads":payloads}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("development","fresh"),required=True)
    parser.add_argument("--confirm-fresh",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.phase=="development":
        seeds=(36900,36901);lrs=(0.003,0.01)
    else:
        if not args.confirm_fresh:
            raise SystemExit("fresh evaluation requires explicit --confirm-fresh after development freeze")
        selection=ROOT/"DEV_SELECTION.json"
        if not selection.exists():
            raise SystemExit("fresh evaluation requires DEV_SELECTION.json from development")
        selected=json.loads(selection.read_text())
        lr=float(selected["selected_learning_rate"])
        if lr not in (0.003,0.01):
            raise SystemExit("selected learning rate is outside the frozen grid")
        seeds=(36910,36911,36912);lrs=(lr,)
    all_rows=[];scores={lr:[] for lr in lrs}
    for seed in seeds:
        for lr in lrs:
            rows,_=run_world(seed,lr,args.phase)
            all_rows.extend(rows)
            if args.phase=="development":
                scores[lr].extend(float(r["nll"]) for r in rows)
            print(json.dumps({"phase":args.phase,"seed":seed,"lr":lr,"rows":len(rows),"mean_nll":sum(float(r["nll"]) for r in rows)/len(rows)},sort_keys=True),flush=True)
    OUT.mkdir(exist_ok=True)
    result_path=OUT/("development.csv" if args.phase=="development" else "fresh.csv")
    with result_path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator="\n");w.writeheader();w.writerows(all_rows)
    if args.phase=="development":
        selected=min(scores,key=lambda lr:sum(scores[lr])/len(scores[lr]))
        (ROOT/"DEV_SELECTION.json").write_text(json.dumps({"experiment_id":"MA-369","selected_learning_rate":selected,"mean_validation_nll_by_lr":{str(k):sum(v)/len(v) for k,v in scores.items()},"development_worlds":list(seeds),"fresh_worlds_locked":[36910,36911,36912],"updates":UPDATES,"correction_updates_per_architecture":CORRECTION_UPDATES},indent=2)+"\n")
        condition="development"
    else:
        condition="fresh"
    core=ROOT/"RESULTS_CORE.csv"
    old=list(csv.DictReader(core.open()))
    with core.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator="\n");w.writeheader();w.writerows(old+all_rows)
    print(json.dumps({"python_seed_count":len(seeds),"methods":len(METHODS),"architectures":len(ARCHS),"rows":len(all_rows),"split":condition},sort_keys=True))

if __name__=="__main__":
    main()
