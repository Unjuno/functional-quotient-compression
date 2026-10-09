#!/usr/bin/env python3
import csv,json,runpy
from pathlib import Path
R=Path(__file__).resolve().parents[1];rows=list(csv.DictReader(open(R/'RESULTS_CORE.csv')));assert len(rows)==8
m=runpy.run_path(str(R/'source'/'run.py'))
for seed in (56701,56702):
 r=m['run'](seed);d={x['method']:x for x in r};assert d['mirror_view']['serialized_bytes']==d['direct_coeff']['serialized_bytes'];assert d['mirror_view']['routed_nmse']==d['direct_coeff']['routed_nmse']
print(json.dumps({'rows':len(rows),'direct_alias':True,'fresh_opened':False,'status':'FAIL'},indent=2))
