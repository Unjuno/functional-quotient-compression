#!/usr/bin/env python3
import csv,json,runpy
from pathlib import Path
R=Path(__file__).resolve().parents[1];rows=list(csv.DictReader(open(R/'RESULTS_CORE.csv')));assert len(rows)==80
m=runpy.run_path(str(R/'source'/'run.py'))
for seed in (58501,58502):assert len(m['run'](seed))==40
print(json.dumps({'rows':len(rows),'attention_metric_replayed':True,'fresh_opened':False,'status':'FAIL storage frontier'},indent=2))
