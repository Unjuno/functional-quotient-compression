# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Export every checkpoint to an equal-overhead FP32 format and verify its forward."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from model import TinyLM
from task import dataset,make_table
from utils import dump
from weights import save_model,load_model

def export_and_verify(train_dir: Path, protocol_path: Path, output: Path):
    p=json.loads(protocol_path.read_text());torch.set_num_threads(p['threads'])
    rows=[]; probe=dataset(4,777,make_table(p['rules'],p['symbols'],p['table_seed']),p['records'])[0]
    for seed in p['seeds']:
        for v in p['variants']:
            stem=f'{v["name"]}_seed{seed}'
            ck=torch.load(train_dir/(stem+'.pt'),weights_only=True,map_location='cpu')
            model=TinyLM(**ck['config']).eval();model.load_state_dict(ck['state_dict'])
            target=output/(stem+'.mnw');save_model(model,target);restored=load_model(target)
            tensor_equal=all(torch.equal(t,restored.state_dict()[k]) for k,t in model.state_dict().items())
            with torch.no_grad():err=(model(probe)[0]-restored(probe)[0]).abs().max().item()
            if not tensor_equal or err!=0:raise AssertionError('roundtrip mismatch')
            rows.append({'experiment':p['experiment'],'name':v['name'],'seed':seed,
                         'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                         'all_parameter_tensors_equal':tensor_equal,'logit_error':err})
    dump(output/'manifest.json',rows)
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--train-dir',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();export_and_verify(a.train_dir,a.protocol,a.output)
