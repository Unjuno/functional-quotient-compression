#!/usr/bin/env python3
import csv,json,runpy
from pathlib import Path
R=Path(__file__).resolve().parents[1];rows=list(csv.DictReader(open(R/'RESULTS_CORE.csv')));assert len(rows)==10
m=runpy.run_path(str(R/'source'/'run.py'))
for seed in (56901,56902):
 r=m['run'](seed);d={x['method']:x for x in r};assert d['mirror_step']['serialized_bytes']==d['direct_time_code']['serialized_bytes'];assert d['mirror_step']['nmse']==d['direct_time_code']['nmse']
print(json.dumps({'rows':len(rows),'direct_alias':True,'fresh_opened':False,'status':'FAIL mirror-specific'},indent=2))
