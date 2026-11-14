# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Independently train selected frozen A6 cases and require byte-identical weights."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import torch
from train import one


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path)
    args=p.parse_args(); torch.set_num_threads(1)
    with (args.out/'train.csv').open() as f: rows=list(csv.DictReader(f))
    selected=[r for r in rows if int(r['world'])==600 and int(r['seed'])==0 and int(r['cap'])==6000]
    if len(selected)!=14: raise ValueError(f'expected 14 selected cases; got {len(selected)}')
    with TemporaryDirectory(prefix='a6_replay_') as tmp:
        results=[]
        for r in selected:
            item=(int(r['world']),r['family'],int(r['seed']),int(r['cap']),r['kind'],700,tmp)
            x=one(item)
            raw=(Path(tmp)/x['weight_file']).read_bytes()
            orig=(args.out/'models'/r['weight_file']).read_bytes()
            if raw!=orig: raise AssertionError('mismatch in '+r['weight_file'])
            results.append({'file':r['weight_file'],'sha256':hashlib.sha256(raw).hexdigest(),'byte_identical':True})
    report={'replayed':len(results),'all_byte_identical':True,'cases':results,
            'boundary':'same runtime/environment, not cross-platform guarantee'}
    (args.out/'REPLAY_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print('REPLAY_OK',len(results),'byte-identical models')

if __name__=='__main__': main()
