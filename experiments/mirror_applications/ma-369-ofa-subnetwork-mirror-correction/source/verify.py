#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import torch
from run import ARCHS, METHODS, make_world, evaluate_bundle, OUT, ROOT

rows=list(csv.DictReader((ROOT/"RESULTS_CORE.csv").open()))
assert rows, "no result rows"
max_diff=0.0
for row in rows:
    seed=int(row["world_or_seed"]); condition=row["condition"]; method=row["method"]
    payload=OUT/"payloads"/condition/str(seed)/row["learning_rate"]/(method+".pt")
    data=payload.read_bytes()
    assert len(data)==int(row["serialized_bytes"])
    assert hashlib.sha256(data).hexdigest()==row["payload_sha256"]
    world=make_world(seed)
    arch_text=row["architecture"]
    width=int(arch_text.split("_")[0][1:]);depth=int(arch_text.split("_")[1][1:]);arch=(width,depth)
    acc,nll,tps,_=evaluate_bundle(data,world,"val" if condition=="development" else "test",arch)
    max_diff=max(max_diff,abs(acc-float(row["accuracy"])),abs(nll-float(row["nll"])))
assert len(rows)%len(ARCHS)==0
print(json.dumps({"rows_replayed":len(rows),"payloads_hash_checked":True,"max_metric_difference":max_diff,"fresh_worlds_present":any(r["condition"]=="fresh" for r in rows)},sort_keys=True))
