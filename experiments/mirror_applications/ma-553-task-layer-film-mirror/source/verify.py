#!/usr/bin/env python3
import csv,json,runpy
from pathlib import Path
R=Path(__file__).resolve().parents[1]; rows=list(csv.DictReader(open(R/'RESULTS_CORE.csv'))); assert len(rows)==12
m=runpy.run_path(str(R/'source'/'run.py'))
for seed in (55301,55302):
 out,b=m['run'](seed); assert b['direct']==b['mirror']
 assert len(out)==6
assert {r['method'] for r in rows}=={'no_modulation','independent_film','flat_table','direct_factorized','mirror_factorized','hyperformer_linear'}
print(json.dumps({'rows':len(rows),'direct_mirror_alias':True,'fresh_opened':False,'status':'FAIL'},indent=2))
