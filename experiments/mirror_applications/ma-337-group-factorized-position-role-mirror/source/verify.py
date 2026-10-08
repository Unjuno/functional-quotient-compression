#!/usr/bin/env python3
import csv,subprocess,sys
from pathlib import Path
EXP=Path(__file__).resolve().parents[1];ROOT=EXP.parents[2];p=EXP/"artifacts"/"results.csv"
before=list(csv.DictReader(p.open()))
subprocess.run([sys.executable,str(EXP/"source"/"run.py")],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
after=list(csv.DictReader(p.open()));assert len(before)==len(after)
cols=[c for c in before[0] if c!="wall_seconds_serialization_eval"]
assert [[r[c] for c in cols] for r in before]==[[r[c] for c in cols] for r in after]
fresh=[r for r in after if int(r["seed"]) in (33711,33712,33713)]
assert len(fresh)==27
for r in fresh:
    if r["method"] in ("factorized_mirror","direct_coordinates","factorized_with_private"):
        assert max(map(float,__import__("json").loads(r["heldout_composition_nMSE"])))<=1e-10
print(f"replay ok: {len(after)} rows; 27 fresh payload/hash/metric rows exact")
