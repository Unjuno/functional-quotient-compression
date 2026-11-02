#!/usr/bin/env python3
import csv,json,runpy
from pathlib import Path
R=Path(__file__).resolve().parents[1];rows=list(csv.DictReader(open(R/'RESULTS_CORE.csv')));assert len(rows)==10
m=runpy.run_path(str(R/'source'/'run.py'))
for seed in (55401,55402):
 r=m['run'](seed);d={x['method']:x for x in r};assert d['mirror_angle']['serialized_bytes']==d['direct_two_basis']['serialized_bytes'];assert d['mirror_angle']['heldout_output_nmse']==d['direct_two_basis']['heldout_output_nmse']
print(json.dumps({'rows':len(rows),'mirror_direct_exact':True,'fresh_opened':False,'status':'FAIL mirror-specific'},indent=2))
