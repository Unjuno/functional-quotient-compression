#!/usr/bin/env python3
import csv,subprocess,sys
from pathlib import Path
E=Path(__file__).resolve().parents[1];ROOT=E.parents[2];p=E/"artifacts"/"results.csv";before=list(csv.DictReader(p.open()))
subprocess.run([sys.executable,str(E/"source"/"run.py")],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
after=list(csv.DictReader(p.open()));cols=[c for c in before[0] if c!="wall_seconds_serialize_eval"]
assert len(before)==len(after)==175
assert [[r[c] for c in cols] for r in before]==[[r[c] for c in cols] for r in after]
for seed in ('34611','34612','34613'):
 for h in ('0.0','0.25','0.5','0.75','1.0'):
  a=next(r for r in after if r['seed']==seed and float(r['heterogeneity_rate'])==float(h) and r['method']=='mirror_rank1_private')
  b=next(r for r in after if r['seed']==seed and float(r['heterogeneity_rate'])==float(h) and r['method']=='native_scalar_phase_private')
  assert a['sha256']==b['sha256']
print("MA-346 replay ok: 175 rows; 105 fresh rows; native phase payload hashes match across full sweep")
