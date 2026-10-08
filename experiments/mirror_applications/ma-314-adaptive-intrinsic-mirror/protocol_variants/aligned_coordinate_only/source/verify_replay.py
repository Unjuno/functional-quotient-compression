from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from engine import METHODS, make_world, make_state, save_payload, load_payload, evaluate, decode_task, normalized_error

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts"
MAX_DIFF=0.0
checked=0
rows=0
for fn in (ROOT/"DEV_RESULTS.csv",ROOT/"FRESH_RESULTS.csv"):
    if not fn.exists(): continue
    with fn.open(newline="") as f: table=list(csv.DictReader(f))
    cache={}
    for row in table:
        seed=int(row["seed"]); method=row["method"]
        if seed not in cache:
            theta0,basis,radii,dims,targets,xtr,xval,xte,ytr,yval,yte=make_world(seed)
            cache[seed]=(theta0,basis,radii,dims,xtr,xval,xte,ytr,yval,yte)
        theta0,basis,radii,dims,xtr,xval,xte,ytr,yval,yte=cache[seed]
        state,ops,_=make_state(method,theta0,basis,radii,xtr,ytr,xval,yval)
        path=OUT/f"{seed}_{method}.npz"
        assert path.exists()
        assert int(row["serialized_bytes"])==path.stat().st_size
        assert row["payload_sha256"]==hashlib.sha256(path.read_bytes()).hexdigest()
        loaded=load_payload(path)
        fresh_path=OUT/f"verify_{seed}_{method}.npz"
        assert save_payload(fresh_path,state)==path.stat().st_size
        assert fresh_path.read_bytes()==path.read_bytes()
        fresh_path.unlink()
        values=np.array([normalized_error(xte[t],yte[t],decode_task(method,loaded,t))[1] for t in range(len(dims))])
        result=float(np.mean(values))
        diff=abs(result-float(row["normalized_mse"]))
        MAX_DIFF=max(MAX_DIFF,diff); assert diff<1e-12
        rows+=1
    checked+=len(table)
(ROOT/"source"/"replay_verification.json").write_text(json.dumps({"result_rows_replayed":rows,"max_metric_difference":MAX_DIFF,"payload_hashes_checked":rows,"serialization_byte_exact":True},indent=2)+"\n")
print(f"verified {rows} result rows; max metric delta={MAX_DIFF:.3g}")
