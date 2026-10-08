#!/usr/bin/env python3
"""Replay MA-335 outputs and verify deterministic payload/metric records."""
import csv, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
EXP=Path(__file__).resolve().parents[1]
result=EXP/"artifacts"/"results.csv"
before=list(csv.DictReader(result.open()))
subprocess.run([sys.executable,str(EXP/"source"/"run.py")],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
after=list(csv.DictReader(result.open()))
cols=[k for k in before[0] if k!="wall_seconds_serialization_and_eval"]
assert len(before)==len(after)
assert [[r[k] for k in cols] for r in before]==[[r[k] for k in cols] for r in after]
fresh=[r for r in after if int(r["seed"]) in (33520,33521,33522)]
assert len(fresh)==21
assert all(float(r["max_related_nMSE"])<1e-10 for r in fresh if r["method"] in ("group_mirror","direct_irrep_coefficients"))
print(f"replay ok: {len(after)} rows; deterministic bytes, hashes, and metrics; {len(fresh)} confirmatory fresh method rows")
