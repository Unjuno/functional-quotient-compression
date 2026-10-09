#!/usr/bin/env python3
import csv,json,runpy
from pathlib import Path
R=Path(__file__).resolve().parents[1];rows=list(csv.DictReader(open(R/'RESULTS_CORE.csv')));assert len(rows)==8
m=runpy.run_path(str(R/'source'/'run.py'))
for seed in (55801,55802):
 r=m['run'](seed);d={x['method']:x for x in r};assert d['mirror_factorized']['serialized_bytes']==d['direct_factorized']['serialized_bytes'];assert float(d['mirror_factorized']['heldout_nmse'])==0
print(json.dumps({'rows':len(rows),'direct_mirror_alias':True,'fresh_opened':False,'status':'FAIL mirror-specific'},indent=2))
