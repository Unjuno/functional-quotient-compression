#!/usr/bin/env python3
import csv,subprocess,sys
from pathlib import Path
E=Path(__file__).resolve().parents[1];ROOT=E.parents[2];p=E/"artifacts"/"development.csv";before=list(csv.DictReader(p.open()))
subprocess.run([sys.executable,str(E/"source"/"run.py"),"--dev-only"],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
after=list(csv.DictReader(p.open()));cols=[c for c in before[0] if c!="wall_seconds_serialize_inference"]
assert len(before)==len(after)==14
assert [[r[c] for c in cols] for r in before]==[[r[c] for c in cols] for r in after]
for seed in ('34101','34102'):
 a=next(r for r in after if r['seed']==seed and r['method']=='mirror_phase_with_private');b=next(r for r in after if r['seed']==seed and r['method']=='native_scalar_phase_with_private');assert a['sha256']==b['sha256']
print("development replay ok: 14 rows; exact phase-control payload hashes; fresh seeds sealed")
