# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Independent-process selected-checkpoint bytewise reproducibility gate for A7."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import torch
import train


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--world',type=int,default=700)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--cap',type=int,default=6000)
    args=ap.parse_args()
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    torch.set_num_threads(1)
    src=args.out
    config=json.loads((src/'config.json').read_text())
    assert args.world in config['worlds'] and args.seed in config['seeds'] and args.cap in config['caps']
    assert train.digest((train.HERE/'PROTOCOL.md').read_bytes())==config['model_protocol_sha256']
    assert train.digest((train.HERE/'a7.py').read_bytes())==config['implementation_sha256']
    assert train.digest((train.HERE.parent/'A6'/'a6.py').read_bytes())==config['a6_core_sha256']
    allrows=list(csv.DictReader((src/'train.csv').open()))
    selected=[r for r in allrows if int(r['world'])==args.world and int(r['seed'])==args.seed and int(r['cap'])==args.cap]
    target=src/'replay_tmp'
    if target.exists():raise FileExistsError('refuse to overwrite replay data')
    target.mkdir()
    results=[]
    for row in selected:
        params=(int(row['world']),row['family'],int(row['seed']),int(row['cap']),row['kind'],config['steps'],str(target))
        repeated=train.one(params)
        actual=(target/row['weight_file']).read_bytes()
        expected=(src/'models'/row['weight_file']).read_bytes()
        if actual!=expected or hashlib.sha256(actual).hexdigest()!=row['digest']:
            raise AssertionError(f"replay mismatch: {row['weight_file']}")
        results.append({'file':row['weight_file'],'byte_identical':True,'sha256':row['digest']})
    (src/'replay.json').write_text(json.dumps({'independent_process':True,'count':len(results),'rows':results},indent=2)+'\n')
    print('REPLAY_OK',len(results),'checkpoints')


if __name__=='__main__':main()
