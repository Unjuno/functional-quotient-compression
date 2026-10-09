#!/usr/bin/env python3
import csv,json,runpy
from pathlib import Path
R=Path(__file__).resolve().parents[1];rows=list(csv.DictReader(open(R/'RESULTS_CORE.csv')));assert len(rows)==30
m=runpy.run_path(str(R/'source'/'run.py'))
for seed in (55701,55702):
 r=m['run'](seed);a=[x for x in r if x['method']=='mirror_roles'];b=[x for x in r if x['method']=='direct_gains'];assert [x['serialized_bytes'] for x in a]==[x['serialized_bytes'] for x in b]
print(json.dumps({'rows':len(rows),'mirror_direct_equal':True,'fresh_opened':False,'status':'FAIL mirror-specific'},indent=2))
