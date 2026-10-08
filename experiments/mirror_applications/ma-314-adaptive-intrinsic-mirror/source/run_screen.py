from __future__ import annotations

import csv
import json
from pathlib import Path

from engine import run_world

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts"
OUT.mkdir(exist_ok=True)
DEV=(31401,31402,31403)
FRESH=(31411,31412,31413)

def run(seeds, name):
    rows=[]
    for seed in seeds:
        rows.extend(run_world(seed,OUT))
    path=ROOT/f"{name}_RESULTS.csv"
    fields=list(rows[0])
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    return rows

def gates(rows):
    by_seed={s:[r for r in rows if r["seed"]==s] for s in sorted({r["seed"] for r in rows})}
    result=[]
    for seed,world in by_seed.items():
        m=next(r for r in world if r["method"]=="adaptive_mirror")
        c=next(r for r in world if r["method"]=="adaptive_coeff")
        said=next(r for r in world if r["method"]=="adaptive_said")
        checks={"quality":m["normalized_mse"]<=1e-4,
                "dimension":m["dimension_accuracy"]>=.95,
                "bytes_coeff":m["serialized_bytes"]<=.90*c["serialized_bytes"],
                "bytes_said":m["serialized_bytes"]<=.90*said["serialized_bytes"],
                "quality_coeff":m["normalized_mse"]<=c["normalized_mse"]+1e-5}
        result.append({"seed":seed,"checks":checks,"all_pass":all(checks.values())})
    return result

if __name__=="__main__":
    dev=run(DEV,"DEV")
    checks=gates(dev)
    (ROOT/"source"/"dev_gate.json").write_text(json.dumps(checks,indent=2)+"\n")
    if all(x["all_pass"] for x in checks):
        fresh=run(FRESH,"FRESH")
        (ROOT/"source"/"fresh_gate.json").write_text(json.dumps(gates(fresh),indent=2)+"\n")
