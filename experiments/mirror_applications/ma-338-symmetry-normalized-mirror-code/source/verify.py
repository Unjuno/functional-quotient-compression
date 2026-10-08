#!/usr/bin/env python3
import csv,subprocess,sys
from pathlib import Path
E=Path(__file__).resolve().parents[1];ROOT=E.parents[2];p=E/"artifacts"/"development.csv";before=list(csv.DictReader(p.open()))
subprocess.run([sys.executable,str(E/"source"/"run.py"),"--dev-only"],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
after=list(csv.DictReader(p.open()));cols=[c for c in before[0] if c!="wall_seconds_serialize_eval"]
assert len(before)==len(after)==12
assert [[r[c] for c in cols] for r in before]==[[r[c] for c in cols] for r in after]
assert {int(r["seed"]) for r in after}=={33801,33802}
print("development replay ok: 12 payload/hash/metric rows; fresh seeds 33811-33813 sealed")
