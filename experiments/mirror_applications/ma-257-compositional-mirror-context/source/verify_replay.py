#!/usr/bin/env python3
"""Replay selected MA-257 development metrics/payloads; fresh was not opened."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as e

ROOT=Path(__file__).resolve().parents[1]
source=json.loads((ROOT/'source'/'dev_results_amended.json').read_text())
expected=[r for r in source['rows'] if r['support_residues']==1]
actual=[]
for seed in (25701,25702,25703):
    for condition in ('aligned','independent'):
        actual.extend(e.run_world(seed,condition,1))
assert len(expected)==len(actual)==54
ignore={'fit_wall_seconds','inference_examples_per_second'}
for a,b in zip(expected,actual):
    for key in a:
        if key in ignore:
            continue
        assert a[key]==b[key], (a['seed'],a['condition'],a['method'],key,a[key],b[key])
print('54 selected-support development rows replay exactly for metrics, bytes, compute proxy, and payload hashes; timings remeasured')
print('Fresh seeds 25711-25713 remained unopened.')
