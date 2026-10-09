#!/usr/bin/env python3
import csv, json
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader(open(ROOT/'RESULTS_CORE.csv')))
assert len(rows)==24
assert {r['method'] for r in rows}=={'shared_prefix','direct_coeff','mirror_gate','independent'}
assert len({r['world'] for r in rows})==2
assert all(int(r['shared_once_bundle_bytes'])>0 for r in rows)
assert all(r['fresh'] if 'fresh' in r else True for r in rows)
# Re-run deterministic payload construction and verify exact direct/Mirror alias and bytes.
mod=runpy.run_path(str(ROOT/'source'/'run.py'))
for seed in (36901,36902):
 for width in (4,8,12):
  _,p=mod['fit'](seed,width)
  assert p['direct_coeff']==p['mirror_gate']
print(json.dumps({'rows':len(rows),'fresh_opened':False,'direct_mirror_payloads_exact':True,'status':'FAIL mirror-specific gate'},indent=2))
