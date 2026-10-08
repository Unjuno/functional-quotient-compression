#!/usr/bin/env python3
"""Portable read-only integrity validator for self-contained MAT01/MAT02 stage."""
from pathlib import Path
from hashlib import sha256
import json,csv
import math
import sys
base=Path(__file__).resolve().parent
R=base/'results'
status=json.loads((R/'VERIFICATION.json').read_text())
errs=[]
for name,dev_count,fresh_count,dev_seeds,fresh_seeds in [
    ('mat01',72,120,[11,12,13],[101,102,103,104,105]),
    ('mat02',96,160,[21,22,23],[201,202,203,204,205])]:
    for phase,total,seeds in [('dev',dev_count,dev_seeds),('fresh',fresh_count,fresh_seeds)]:
        path=R/name/f'{phase}_raw.csv'
        with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
        if len(rows)!=total:errs.append(f'{name} {phase} rows mismatch')
        if sorted(set(int(r['seed']) for r in rows))!=seeds:errs.append(f'{name} {phase} seeds mismatch')
        if len({(r['seed'],r['method'],r['role']) for r in rows})!=total:errs.append(f'{name} {phase} duplicates')
        if any(not math.isfinite(float(row['nll'])) for row in rows):errs.append(f'{name} {phase} NaN')
for path,claim in {**status['source_hashes'],**status['results_hashes']}.items():
    actual=sha256((base/path).read_bytes()).hexdigest()
    if actual!=claim:errs.append('hash mismatch '+path)
replay=json.loads((R/'REPLAY_VERIFICATION.json').read_text())
for k,v in replay.items():
    if not v['passed'] or v['max_abs_difference']!=0:errs.append('replay failed '+k)
print(json.dumps({'dataset_series':2,'fresh_rows':280,'fresh_world_seeds':10,'source_tests':14,
                  'replay_exact':all(x['passed'] for x in replay.values()),'errors':errs},ensure_ascii=False))
sys.exit(bool(errs))
