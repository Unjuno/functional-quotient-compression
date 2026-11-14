# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Independently repeat selected fixed-seed A8 runs in new Python processes."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import a8


def run_case(out:Path,record:dict)->dict:
    env=a8.make_world(int(record['world']),record['family'])
    model=a8.train_one(env,record['kind'],int(record['cap']),int(record['seed']),
                       steps=int(json.loads((out/'config.json').read_text())['steps']))['model']
    actual=hashlib.sha256(a8.serialize(model)).hexdigest()
    return {'weight_file':record['weight_file'],'recorded':record['sha256'],
            'reproduced':actual,'byte_identical':actual==record['sha256']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case',type=int,default=None)
    a=p.parse_args()
    rows=list(csv.DictReader((a.out/'train.csv').open()))
    if a.case is not None:
        print(json.dumps(run_case(a.out,rows[a.case])))
        return
    # Explicitly sample across conditions and use new child processes.
    wanted=[0,len(rows)//5,len(rows)//2,(4*len(rows))//5,len(rows)-1]
    if len(rows)<100:wanted=[0,len(rows)//2,len(rows)-1]
    results=[]
    for i in sorted(set(wanted)):
        proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--out',str(a.out),'--case',str(i)],
                             check=True,capture_output=True,text=True,
                             env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        row=json.loads(proc.stdout)
        if not row['byte_identical']:raise RuntimeError(f"replay mismatch: {row['weight_file']}")
        results.append(row)
    evidence={'schema':'a8-replay-v1','subprocess_runs':len(results),'all_byte_identical':True,'rows':results}
    (a.out/'replay.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('A8_REPLAY_PASSED',len(results))

if __name__=='__main__':main()
