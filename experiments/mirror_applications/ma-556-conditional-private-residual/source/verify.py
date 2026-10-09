#!/usr/bin/env python3
import csv,json,runpy
from pathlib import Path
R=Path(__file__).resolve().parents[1];rows=list(csv.DictReader(open(R/'RESULTS_CORE.csv')));assert len(rows)==80
m=runpy.run_path(str(R/'source'/'run.py'))
for seed in (55601,55602):
 r=m['run'](seed);assert len(r)==40
 assert all(int(x['serialized_bytes'])>0 for x in r)
print(json.dumps({'rows':len(rows),'fresh_opened':False,'status':'FAIL storage frontier'},indent=2))
