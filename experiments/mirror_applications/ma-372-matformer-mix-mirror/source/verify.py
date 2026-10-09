#!/usr/bin/env python3
import csv,json,runpy
from pathlib import Path
R=Path(__file__).resolve().parents[1]; rows=list(csv.DictReader(open(R/'RESULTS_CORE.csv'))); assert len(rows)==8
m=runpy.run_path(str(R/'source'/'run.py'))
for seed in (37201,37202):
 out,s=m['run'](seed);assert s['mirror']==s['direct']
assert len(m['HOLD'])==6
print(json.dumps({'rows':len(rows),'direct_mirror_exact':True,'fresh_opened':False,'status':'FAIL Mirror-specific'},indent=2))
