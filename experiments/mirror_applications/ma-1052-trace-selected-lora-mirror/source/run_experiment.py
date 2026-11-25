"""MA-1052 faithful TRACE-style CPU screen on MOMENT-1-base.

Implements gated rank-2 LoRA, masked validation-gradient DSIC selection, and
the TRACE ProjDown head. The code is paper-derived because no official TRACE
repository was located. Audit is loaded only after a separate run invocation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CONTEXT, PATCH, REDUCED, HORIZONS = 512, 8, 96, (192, 720)
MODULES = ("q", "k", "v", "o", "wi_0", "wi_1", "wo")
DATASETS = ("ETTh1", "ETTh2")
ROLES = tuple((d, h) for d in DATASETS for h in HORIZONS)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


class GatedLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, n_roles: int, view: str, rank: int = 2):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.normal_(self.b.weight, mean=0.0, std=0.01)
        self.gate = nn.Parameter(torch.ones(()))
        self.register_buffer("selected", torch.tensor(True))
        self.view = view
        self.active_role = 0
        self.code = nn.Parameter(torch.zeros(n_roles, rank)) if view in ("mirror", "diag") else None

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        if not bool(self.selected):
            return self.base(x)
        z = self.a(x)
        if self.code is not None:
            c = self.code[self.active_role]
            if self.view == "mirror":
                theta = math.pi * torch.tanh(c[0])
                gain = torch.exp(torch.tanh(c[1]))
                c0, c1 = torch.cos(theta), torch.sin(theta)
                z = torch.stack((c0 * z[..., 0] - c1 * z[..., 1], c1 * z[..., 0] + c0 * z[..., 1]), dim=-1) * gain
            else:
                z = z * torch.exp(torch.tanh(c))
        return self.base(x) + self.gate * self.b(z)


class ProjDownHead(nn.Module):
    """TRACE Appendix-A projection down (d/β=96) then forecast projection."""
    def __init__(self, horizon: int):
        super().__init__()
        self.down = nn.Linear(768, REDUCED)
        self.out = nn.Linear((CONTEXT // PATCH) * REDUCED, horizon)

    def forward(self, x, input_mask=None):
        b, c, p, d = x.shape
        x = self.down(x).reshape(b, c, p * REDUCED)
        return self.out(x)


class LinearProbeHead(nn.Module):
    def __init__(self, horizon: int):
        super().__init__()
        self.out = nn.Linear((CONTEXT // PATCH) * 768, horizon)

    def forward(self, x, input_mask=None):
        b, c, p, d = x.shape
        return self.out(x.reshape(b, c, p * d))


def _replace_linear(model, mode: str, n_roles: int):
    wrapped = {}
    for name, module in list(model.encoder.named_modules()):
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in MODULES:
            parent_name, _, leaf = name.rpartition(".")
            parent = model.encoder.get_submodule(parent_name) if parent_name else model.encoder
            parent._modules[leaf] = GatedLoRALinear(module, n_roles, mode)
            wrapped[name] = parent._modules[leaf]
    if len(wrapped) != 84:
        raise RuntimeError(f"TRACE expected 84 rank-2 modules, found {len(wrapped)}")
    return wrapped


def load_model(checkpoint: str, view: str, horizon: int, roles: int = len(ROLES), linear_probe=False):
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(
        checkpoint,
        model_kwargs={
            "task_name": "forecasting", "forecast_horizon": horizon,
            "freeze_encoder": True, "freeze_embedder": True, "freeze_head": False,
            "enable_gradient_checkpointing": False,
        },
        local_files_only=True,
    )
    model.init()
    if hasattr(model.encoder, "gradient_checkpointing_disable"):
        model.encoder.gradient_checkpointing_disable()
    for p in model.parameters():
        p.requires_grad_(False)
    adapters = {} if linear_probe else _replace_linear(model, view, roles)
    if linear_probe:
        model.head = LinearProbeHead(horizon)
    else:
        model.head = ProjDownHead(horizon)
        for p in model.head.parameters():
            p.requires_grad_(True)
    if linear_probe:
        for p in model.head.parameters():
            p.requires_grad_(True)
    return model, adapters


def load_series():
    series, split = {}, {}
    for ds in DATASETS:
        df = pd.read_csv(ROOT / "runs/raw" / f"{ds}.csv")
        vals = df["OT"].to_numpy(np.float64)
        n_train, n_val, n_test = 12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24
        mu, sd = float(vals[:n_train].mean()), float(vals[:n_train].std())
        z = (vals - mu) / sd
        series[ds] = z.astype(np.float32)
        split[ds] = {"train": n_train, "val": n_train + n_val, "test": n_train + n_val + n_test,
                     "mase_scale": float(np.abs(z[24:n_train] - z[:n_train-24]).mean()),
                     "mean": mu, "std": sd, "raw_length": int(len(vals))}
    return series, split


def role_data(series, split, role, run_split):
    ds, horizon = role
    arr, b = series[ds], split[ds]
    if run_split == "train":
        end = b["train"] - CONTEXT - horizon
        starts = np.arange(max(0, end + 1), dtype=np.int64)
    else:
        lo = b["train"] if run_split == "dev" else b["val"]
        hi = b["val"] if run_split == "dev" else b["test"]
        # Origins denote first target index; use evenly spaced, horizon-nonoverlapping audit windows.
        # Development origins match the executed screen: approximately eight
        # windows per role, allowing overlap for H=720; test origins are disjoint.
        stride = max(1, (hi - lo) // 8) if run_split == "dev" else horizon
        starts = np.arange(lo, hi - horizon + 1, stride, dtype=np.int64)
        if run_split == "dev" and len(starts) > 8:
            starts = starts[np.linspace(0, len(starts)-1, 8, dtype=int)]
    x, y = [], []
    for target_start in starts:
        x0 = int(target_start - CONTEXT) if run_split != "train" else int(target_start)
        y0 = int(target_start) + (CONTEXT if run_split == "train" else 0)
        # Training starts are context origins; evaluation starts are target origins.
        if run_split == "train":
            y0 = x0 + CONTEXT
        xx = arr[x0:x0 + CONTEXT]
        yy = arr[y0:y0 + horizon]
        if len(xx) == CONTEXT and len(yy) == horizon:
            x.append(xx); y.append(yy)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), b["mase_scale"], starts


def batch_from(arrays, batch, rng):
    x, y, _, starts = arrays
    idx = rng.randint(0, len(x), size=batch)
    return torch.from_numpy(x[idx, None, :]), torch.from_numpy(y[idx, None, :])


def set_role(adapters, role_idx):
    for m in adapters.values():
        m.active_role = role_idx


def train_step(model, adapters, roles, data, optimizer, rngs):
    model.train(); optimizer.zero_grad(set_to_none=True)
    for role_idx, role in enumerate(roles):
        set_role(adapters, role_idx)
        x, y = batch_from(data[role], 4, rngs[role_idx])
        pred = model(x_enc=x, input_mask=torch.ones(4, CONTEXT)).forecast[..., :role[1]]
        ((pred - y).square().mean() / len(roles)).backward()
    optimizer.step()


def dsic_select(model, adapters, roles, train_data, val_data, seed, role_index=None):
    """Twelve updates in total: two warmups + ten masked-gradient prune rounds."""
    dsic_started = time.perf_counter()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=2e-4)
    rngs = [np.random.RandomState(seed + 1000 + i) for i in range(len(roles))]
    # Joint pooled warm-up (or role-only specialist warm-up).
    for _ in range(2):
        train_step(model, adapters, roles, train_data, opt, rngs)
    active = list(adapters)
    v_rngs = [np.random.RandomState(seed + 5000 + i) for i in range(len(roles))]
    for _round in range(10):
        sums = {n: 0.0 for n in active}; counts = {n: 0 for n in active}
        for _ in range(8):
            masked = set(random.Random(seed + _round * 101 + _).sample(active, max(1, round(0.1 * len(active)))))
            for n, module in adapters.items():
                module.selected.fill_(n in active and n not in masked)
            model.train()
            for p in trainable: p.grad = None
            use_roles = roles if role_index is None else [roles[0]]
            for rr, role in enumerate(use_roles):
                actual_idx = role_index if role_index is not None else rr
                set_role(adapters, actual_idx)
                x, y = batch_from(val_data[role], 2, v_rngs[actual_idx])
                pred = model(x_enc=x, input_mask=torch.ones(2, CONTEXT)).forecast[..., :role[1]]
                (pred - y).square().mean().backward()
            for n in active:
                g = adapters[n].gate.grad
                if n not in masked and g is not None:
                    sums[n] += float(g.detach().abs()); counts[n] += 1
        scores = {n: sums[n] / max(1, counts[n]) for n in active}
        remove = sorted(active, key=lambda n: (scores[n], n))[:8]
        active = [n for n in active if n not in set(remove)]
        for n, module in adapters.items():
            module.selected.fill_(n in active)
        # One optimizer update between pruning rounds, as in TRACE's staged schedule.
        train_step(model, adapters, roles, train_data, opt, rngs)
    assert len(active) == 4
    return active, opt, 12, time.perf_counter() - dsic_started


def finish_train(model, adapters, roles, data, optimizer, rngs, steps=18):
    for _ in range(steps):
        train_step(model, adapters, roles, data, optimizer, rngs)


def eval_model(model, adapters, roles, data, run_split):
    model.eval(); results = {}
    with torch.no_grad():
        for i, role in enumerate(roles):
            set_role(adapters, i)
            x, y, scale, starts = data[role]
            pred = model(x_enc=torch.from_numpy(x[:, None, :]), input_mask=torch.ones(len(x), CONTEXT)).forecast[..., :role[1]].numpy()[:, 0]
            tgt = y
            mae = np.abs(pred - tgt).mean(axis=1)
            mase = mae / scale
            smape = (2 * np.abs(pred-tgt) / (np.abs(pred)+np.abs(tgt)+1e-6)).mean(axis=1)
            results[f"{role[0]}_H{role[1]}"] = {"mase": float(mase.mean()), "smape": float(smape.mean()), "n_origins": int(len(x)), "origins": [int(v) for v in starts]}
    results["macro_mase"] = float(np.mean([v["mase"] for k,v in results.items() if isinstance(v,dict)]))
    results["macro_smape"] = float(np.mean([v["smape"] for k,v in results.items() if isinstance(v,dict)]))
    return results


def latency_profile(model, adapters, roles):
    model.eval(); samples=[]
    with torch.no_grad():
        for i,_role in enumerate(roles):
            set_role(adapters,i)
            x=torch.zeros(1,1,CONTEXT); mask=torch.ones(1,CONTEXT)
            model(x_enc=x,input_mask=mask)
            for _ in range(5):
                t=time.perf_counter(); model(x_enc=x,input_mask=mask); samples.append(time.perf_counter()-t)
    vals=np.asarray(samples)
    return {"n_single_example_runs":int(len(vals)),"p50_seconds":float(np.percentile(vals,50)),"p95_seconds":float(np.percentile(vals,95)),"examples_per_second_at_p50":float(1/np.percentile(vals,50))}


def mac_proxy(selected, adapters, horizon, mirror_ops=0):
    # Dense T5 encoder, sequence length 64; linear projections, gated FFN, and QK/AV.
    n=CONTEXT//PATCH; d=768; ff=2048; layers=12
    backbone=layers*(4*n*d*d + 3*n*d*ff + 2*n*n*d)
    lora=sum(2*(adapters[name].in_features+adapters[name].out_features)*n*layers for name in selected)
    head=n*d*REDUCED+n*REDUCED*horizon
    return {"approx_backbone_macs_per_example":int(backbone),"selected_lora_macs_per_example":int(lora),"projdown_head_macs_per_example":int(head),"mirror_elementary_ops_per_example":int(mirror_ops)}


def save_selected(model, adapters, selected, outdir, method, role_codes=False, splits=None):
    from safetensors.torch import save_file
    state = {}
    for n in selected:
        m = adapters[n]
        state[n + ".A"] = m.a.weight.detach().cpu().contiguous()
        state[n + ".B"] = m.b.weight.detach().cpu().contiguous()
        state[n + ".gate"] = m.gate.detach().cpu().reshape(1).contiguous()
        if role_codes and m.code is not None:
            state[n + ".view_code"] = m.code.detach().cpu().contiguous()
    for n, p in model.head.state_dict().items():
        state["head." + n] = p.detach().cpu().contiguous()
    outdir.mkdir(parents=True, exist_ok=True)
    weight_file = outdir / "adapter.safetensors"
    save_file(state, str(weight_file))
    meta = {"method": method, "selected_modules": selected, "role_codes": role_codes,
            "roles": [f"{d}_H{h}" for d,h in ROLES], "base_model": "AutonLab/MOMENT-1-base",
            "base_revision": "5e44b0ea26376a176360f87831124e018f876d96", "head": "TRACE ProjDown beta=8",
            "preprocessing": {d:{"mean":splits[d]["mean"],"std":splits[d]["std"]} for d in DATASETS} if splits else None}
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"weights_bytes": weight_file.stat().st_size, "metadata_bytes": (outdir / "metadata.json").stat().st_size,
            "adaptation_bytes": weight_file.stat().st_size + (outdir / "metadata.json").stat().st_size}


def forecast_payload(model, adapters, roles, data):
    return eval_model(model, adapters, roles, data, "eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=105201)
    ap.add_argument("--evaluation", choices=("dev", "audit"), default="dev")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    set_seed(args.seed)
    started = time.perf_counter()
    series, splits = load_series()
    train_data = {r: role_data(series, splits, r, "train") for r in ROLES}
    val_data = {r: role_data(series, splits, r, "dev") for r in ROLES}
    eval_data = {r: role_data(series, splits, r, args.evaluation) for r in ROLES}
    rngs = [np.random.RandomState(args.seed + i) for i in range(len(ROLES))]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    run = {"experiment_id":"MA-1052", "seed":args.seed, "evaluation":args.evaluation,
           "roles":[f"{d}_H{h}" for d,h in ROLES], "base_revision":"5e44b0ea26376a176360f87831124e018f876d96",
           "training_updates_per_role":30, "batch_size_per_role":4, "methods":{}, "wall_start_unix":time.time()}

    # Pooled TRACE basis and single DSIC selection, reused as the common initialization for the three pooled arms.
    shared, adapters = load_model(args.checkpoint, "shared", 720)
    selected, opt, dsic_updates, dsic_wall = dsic_select(shared, adapters, ROLES, train_data, val_data, args.seed)
    initial = {"lora": {n: {"a": adapters[n].a.state_dict(), "b": adapters[n].b.state_dict(), "gate": adapters[n].gate.detach().clone()} for n in selected},
               "head": copy.deepcopy(shared.head.state_dict()), "selected": selected}
    del shared, adapters, opt
    for view, name in (("shared","pooled_trace_shared"),("diag","pooled_trace_diag"),("mirror","pooled_trace_mirror")):
        model, mods = load_model(args.checkpoint, view, 720)
        for n in selected:
            mods[n].a.load_state_dict(initial["lora"][n]["a"])
            mods[n].b.load_state_dict(initial["lora"][n]["b"])
            mods[n].gate.data.copy_(initial["lora"][n]["gate"])
            mods[n].selected.fill_(True)
        model.head.load_state_dict(initial["head"])
        # Train only selected modules and forecast head; newly added codes start at identity.
        for n, m in mods.items():
            m.selected.fill_(n in selected)
            for p in m.parameters(): p.requires_grad_(n in selected)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=2e-4)
        local_rngs = [np.random.RandomState(args.seed + 10000 + i) for i in range(len(ROLES))]
        t = time.perf_counter(); finish_train(model, mods, ROLES, train_data, optimizer, local_rngs, 18); train_s=time.perf_counter()-t
        metrics = eval_model(model, mods, ROLES, eval_data, args.evaluation)
        payload = save_selected(model, mods, selected, out / name, name, view in ("mirror","diag"), splits)
        lat=latency_profile(model,mods,ROLES)
        run["methods"][name] = {"metrics":metrics,"payload":payload,"selected":selected,"train_updates":18,"dsic_shared_updates_amortized":dsic_updates,"dsic_validation_examples_amortized":640,"dsic_wall_s_amortized":dsic_wall,"training_examples":480,"train_wall_s":train_s,"inference_latency":lat,"active_compute_proxy_by_role":{f"{d}_H{h}":mac_proxy(selected,mods,h,1536 if view=="mirror" else (512 if view=="diag" else 0)) for d,h in ROLES}}
        del model, mods, optimizer

    # Native MOMENT linear-probe specialists (frozen encoder, conventional forecast heads).
    probe_metrics = {}; probe_payload = 0; probe_wall = 0.0
    from safetensors.torch import save_file
    for role_idx, role in enumerate(ROLES):
        name = f"moment_linear_probe_{role[0]}_H{role[1]}"
        model, _ = load_model(args.checkpoint, "shared", role[1], roles=1, linear_probe=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
        local_rngs = [np.random.RandomState(args.seed + 25000 + role_idx)]
        t=time.perf_counter()
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            x,y=batch_from(train_data[role],4,local_rngs[0])
            pred=model(x_enc=x,input_mask=torch.ones(4,CONTEXT)).forecast
            (pred-y).square().mean().backward(); optimizer.step()
        train_s=time.perf_counter()-t
        met=eval_model(model,{},[role],{role:eval_data[role]},args.evaluation)
        dest=out/name; dest.mkdir(parents=True,exist_ok=True)
        weights=dest/"head.safetensors"; save_file({k:v.detach().cpu().contiguous() for k,v in model.head.state_dict().items()},str(weights))
        metadata={"method":"MOMENT linear probe","role":f"{role[0]}_H{role[1]}","base_revision":"5e44b0ea26376a176360f87831124e018f876d96","preprocessing":{"mean":splits[role[0]]["mean"],"std":splits[role[0]]["std"]}}
        mf=dest/"metadata.json"; mf.write_text(json.dumps(metadata,indent=2)+"\n")
        b=weights.stat().st_size+mf.stat().st_size
        probe_payload+=b; probe_wall+=train_s
        probe_metrics.update({k:v for k,v in met.items() if k!="macro_mase"})
        lat=latency_profile(model,{},[role])
        run["methods"][name]={"metrics":met,"adaptation_bytes":b,"train_updates":30,"training_examples":120,"train_wall_s":train_s,"inference_latency":lat,"active_compute_proxy":{"approx_backbone_macs_per_example":int(12*(4*64*768*768+3*64*768*2048+2*64*64*768)),"linear_probe_head_macs_per_example":int(64*768*role[1])}}
        del model,optimizer
    run["methods"]["moment_linear_probe_reference"]={"metrics":probe_metrics,"macro_mase":float(np.mean([v["mase"] for v in probe_metrics.values() if isinstance(v,dict)])),"adaptation_bytes_sum":probe_payload,"train_wall_s_sum":probe_wall}

    # Independent native TRACE specialists. Each has task-specific Gated DSIC selection and ProjDown head.
    specialist_metrics = {}; specialist_payload = 0; specialist_wall = 0.0
    all_selected = {}
    for role_idx, role in enumerate(ROLES):
        name = f"specialist_{role[0]}_H{role[1]}"
        model, mods = load_model(args.checkpoint, "shared", role[1], roles=1)
        selected_i, optimizer, dsic_i, dsic_wall_i = dsic_select(model, mods, [role], train_data, val_data, args.seed + 300 + role_idx, role_index=0)
        local_rngs = [np.random.RandomState(args.seed + 20000 + role_idx)]
        t=time.perf_counter(); finish_train(model, mods, [role], train_data, optimizer, local_rngs, 18); train_s=time.perf_counter()-t
        met=eval_model(model, mods, [role], {role:eval_data[role]}, args.evaluation)
        payload=save_selected(model,mods,selected_i,out/name,name,False,splits)
        specialist_metrics.update({k:v for k,v in met.items() if k!="macro_mase"})
        specialist_payload += payload["adaptation_bytes"]
        specialist_wall += train_s
        all_selected[name]=selected_i
        lat=latency_profile(model,mods,[role])
        run["methods"][name]={"metrics":met,"payload":payload,"selected":selected_i,"train_updates":30,"dsic_updates":dsic_i,"dsic_validation_examples":160,"dsic_wall_s":dsic_wall_i,"training_examples":120,"train_wall_s":train_s,"inference_latency":lat,"active_compute_proxy_by_role":{f"{role[0]}_H{role[1]}":mac_proxy(selected_i,mods,role[1])}}
        del model,mods,optimizer
    specialist_macro=float(np.mean([v["mase"] for v in specialist_metrics.values() if isinstance(v,dict)]))
    run["methods"]["independent_trace_reference"]={"metrics":specialist_metrics,"macro_mase":specialist_macro,"adaptation_bytes_sum":specialist_payload,"train_wall_s_sum":specialist_wall,"selected_by_role":all_selected}

    # Seasonal-naive secondary reference, no trainable payload.
    naive={}
    for role in ROLES:
        arr, b = series[role[0]], splits[role[0]]
        x,y,scale,starts=eval_data[role]
        # Seasonal period 24, repeating the last day of context.
        pred=np.stack([arr[int(t)-24:int(t)] for t in starts])
        pred=np.resize(pred,(len(starts),role[1]))
        mase=(np.abs(pred-y).mean(axis=1)/scale).mean()
        naive[f"{role[0]}_H{role[1]}"]={"mase":float(mase),"n_origins":int(len(y))}
    naive["macro_mase"]=float(np.mean([v["mase"] for k,v in naive.items() if isinstance(v,dict)]))
    run["methods"]["seasonal_naive"]=naive
    run["elapsed_wall_s"]=time.perf_counter()-started
    run["data"]={ds:{**splits[ds],"sha256":hashlib.sha256((ROOT/"runs/raw"/f"{ds}.csv").read_bytes()).hexdigest()} for ds in DATASETS}
    (out/"run.json").write_text(json.dumps(run,indent=2)+"\n")
    summaries={}
    for k,v in run["methods"].items():
        if not isinstance(v,dict): summaries[k]=v; continue
        summaries[k]=v.get("macro_mase",(v.get("metrics") or {}).get("macro_mase"))
    print(json.dumps({"evaluation":args.evaluation,"elapsed_wall_s":run["elapsed_wall_s"],"methods":summaries},indent=2))


if __name__ == "__main__":
    main()
