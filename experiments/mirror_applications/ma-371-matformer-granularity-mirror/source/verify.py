#!/usr/bin/env python3
import csv,json,runpy
from pathlib import Path
R=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader(open(R/'RESULTS_CORE.csv')))
assert len(rows)==48
assert len({r['world'] for r in rows})==2
assert {r['method'] for r in rows}=={'shared_prefix','direct_coeff','mirror_gate','independent'}
mod=runpy.run_path(str(R/'source'/'run.py'))
for seed in (37101,37102):
 _,s=mod['runworld'](seed)
 assert s['direct']==s['mirror']
print(json.dumps({'rows':len(rows),'direct_mirror_bytes_equal':True,'fresh_opened':False,'status':'FAIL Mirror-specific gate'},indent=2))
