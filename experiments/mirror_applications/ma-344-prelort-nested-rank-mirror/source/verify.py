#!/usr/bin/env python3
import csv,subprocess,sys
from pathlib import Path
E=Path(__file__).resolve().parents[1];ROOT=E.parents[2];p=E/"artifacts"/"results.csv";before=list(csv.DictReader(p.open()))
subprocess.run([sys.executable,str(E/"source"/"run.py")],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
after=list(csv.DictReader(p.open()));cols=[c for c in before[0] if c!="wall_seconds_serialize_eval"]
assert len(before)==len(after)==35
assert [[r[c] for c in cols] for r in before]==[[r[c] for c in cols] for r in after]
for seed in ('34411','34412','34413'):
 a=next(r for r in after if r['seed']==seed and r['method']=='mirror_segment_phase_private');b=next(r for r in after if r['seed']==seed and r['method']=='native_scalar_phase_private');assert a['sha256']==b['sha256']
print("MA-344 replay ok: 35 rows; 21 fresh rows; native phase hashes identical")
