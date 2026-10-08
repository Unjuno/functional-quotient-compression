"""Development and fresh screen for the frozen MA-504 mechanism protocol."""
from __future__ import annotations
import csv, hashlib, json, math, statistics, time
from pathlib import Path

import torch
from torch.nn import functional as F

from model import Intervention, make_batch, save_payload

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "source" / "artifacts"
OUT.mkdir(exist_ok=True)
MODES = ("static", "independent", "mirror", "film", "full")
SEEDS = (50401, 50402)
FRESH = (50403, 50404, 50405)
UPDATES, BATCH = 700, 64
LRS = (0.001, 0.003)


def fit_and_eval(seed, mode, lr, split, phase):
    torch.manual_seed(seed + MODES.index(mode) * 100 + int(lr * 1_000_000))
    model = Intervention(mode, seed + MODES.index(mode) * 17)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    train_x, train_c, train_y = make_batch(seed, "train", 32768)
    gen = torch.Generator(device="cpu").manual_seed(seed + 55_000 + MODES.index(mode))
    model.train(); start = time.perf_counter()
    for _ in range(UPDATES):
        ix = torch.randint(train_x.shape[0], (BATCH,), generator=gen)
        pred = model(train_x[ix], train_c[ix])
        loss = F.mse_loss(pred, train_y[ix])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    elapsed = time.perf_counter() - start
    x, context, y = make_batch(seed, split, 8192)
    model.eval()
    with torch.no_grad():
        pred = model(x, context)
        mse = float(F.mse_loss(pred, y))
        r2 = float(1 - (pred-y).square().sum() / (y-y.mean()).square().sum())
        for _ in range(2): model(x[:256], context[:256])
        t = time.perf_counter()
        for _ in range(8): model(x[:256], context[:256])
        throughput = 256 * 8 / (time.perf_counter() - t)
    lr_tag = str(lr).replace(".", "")
    path = OUT / f"{phase}_seed{seed}_{mode}_lr{lr_tag}.pt"
    payload = save_payload(path, model, seed)
    payload["path"] = str(path.relative_to(ROOT))
    return model, {"seed": seed, "condition": phase, "mode": mode, "lr": lr,
        "mse": mse, "r2": r2, "wall_seconds": elapsed,
        "examples_seen": UPDATES * BATCH, "updates": UPDATES,
        "active_macs_per_token": model.active_macs_per_token(),
        "throughput_examples_per_second": throughput,
        "payload": payload}


def gate(rows, seeds):
    by={(r["seed"],r["mode"]):r for r in rows}
    checks=[]
    for seed in seeds:
        m, f, i = by[(seed,"mirror")], by[(seed,"film")], by[(seed,"independent")]
        matched=abs(m["payload"]["bytes"]-f["payload"]["bytes"]) <= .05*m["payload"]["bytes"]
        checks.append({"seed":seed,
            "mirror_vs_independent_quality":m["mse"] <= 1.10*i["mse"],
            "mirror_vs_independent_bytes":m["payload"]["bytes"] <= .80*i["payload"]["bytes"],
            "byte_match_film":matched,
            "mirror_beats_film_2pct":matched and m["mse"] <= .98*f["mse"]})
    return {"checks":checks,"pass":all(all(v for k,v in c.items() if k!="seed") for c in checks)}


def write_rows(rows):
    path=ROOT/"RESULTS_CORE.csv"
    fields=["condition","world_or_seed","method","serialized_bytes","adapter_only_bytes","train_tokens_or_examples","optimizer_updates","active_compute_proxy","wall_time_s","primary_metric","primary_value","secondary_metric","secondary_value","status_note"]
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader()
        for r in rows:
            w.writerow({"condition":r["condition"],"world_or_seed":r["seed"],"method":"MA-504 synthetic ReFT","serialized_bytes":r["payload"]["bytes"],"adapter_only_bytes":r["payload"]["bytes"],"train_tokens_or_examples":r["examples_seen"],"optimizer_updates":r["updates"],"active_compute_proxy":r["active_macs_per_token"],"wall_time_s":r["wall_seconds"],"primary_metric":"held-out token-output MSE","primary_value":r["mse"],"secondary_metric":"R2;examples/s","secondary_value":f"{r['r2']:.9g};{r['throughput_examples_per_second']:.6f}","status_note":f"mode={r['mode']};lr={r['lr']}"})


def main():
    torch.set_num_threads(1)
    dev=[]; lr_scores={lr:[] for lr in LRS}; dev_by={}
    for seed in SEEDS:
        for lr in LRS:
            for mode in MODES:
                _,row=fit_and_eval(seed,mode,lr,"dev","development")
                dev.append(row);lr_scores[lr].append(row["mse"])
                dev_by[(seed,lr,mode)]=row
                print("dev",seed,lr,mode,row["mse"],row["payload"]["bytes"],flush=True)
    # Select a common rate by mean rank-free pooled MSE, fixed before fresh tasks.
    for lr in LRS:
        lr_scores[lr]=[dev_by[(seed,lr,"mirror")]["mse"]-dev_by[(seed,lr,"film")]["mse"] for seed in SEEDS]
    selected=min(LRS,key=lambda lr:(statistics.mean(lr_scores[lr]),lr))
    fresh=[]
    for seed in FRESH:
        for mode in MODES:
            _,row=fit_and_eval(seed,mode,selected,"fresh","fresh")
            fresh.append(row)
            print("fresh",seed,selected,mode,row["mse"],row["payload"]["bytes"],flush=True)
    fresh_gate=gate(fresh,FRESH)
    audit_opened=bool(fresh_gate["pass"])
    audit_rows=[]
    if audit_opened:
        # The locked audit generator is only invoked after the complete fresh gate passes.
        for seed in FRESH:
            for mode in ("mirror","film","independent"):
                artifact=OUT/f"fresh_seed{seed}_{mode}_lr{str(selected).replace('.', '')}.pt"
                payload=torch.load(artifact,map_location="cpu",weights_only=False)
                model=Intervention(mode,seed+MODES.index(mode)*17)
                model.load_state_dict(payload["state_dict"])
                x,c,y=make_batch(seed,"audit",8192)
                with torch.no_grad():mse=float(F.mse_loss(model(x,c),y))
                audit_rows.append({"seed":seed,"mode":mode,"mse":mse})
    all_rows=dev+fresh
    write_rows(all_rows)
    summary={"development":dev,"development_paired_mirror_minus_film":{str(k):statistics.mean(v) for k,v in lr_scores.items()},"selected_learning_rate":selected,
             "fresh":fresh,"fresh_gate":fresh_gate,"audit_opened":audit_opened,"audit":audit_rows,
             "protocol":{"seeds":SEEDS,"fresh_seeds":FRESH,"updates":UPDATES,"batch":BATCH}}
    (ROOT/"source"/"screen_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"selected_lr":selected,"fresh_gate":fresh_gate,"audit_opened":audit_opened},indent=2),flush=True)


if __name__=="__main__":main()
