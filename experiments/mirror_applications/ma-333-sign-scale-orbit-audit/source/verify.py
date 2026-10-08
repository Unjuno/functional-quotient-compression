import hashlib,json
import numpy as np
from pathlib import Path
import torch
import run
ROOT=Path(__file__).resolve().parents[1];checked=0;max_fp32=0.0
for split in ('development_amended','fresh'):
 folder=ROOT/'artifacts'/split
 for rp in sorted(folder.glob('*.json')):
  d=json.loads(rp.read_text());seed=d['seed'];condition=d['condition']
  for act in ('relu','tanh'):
   model,x,_=run.world(seed,act);basepath=folder/f'{condition}_{seed}_{act}_base.zip';base_raw=basepath.read_bytes();base=run.load(basepath)
   ref=run.eval_arrays(base,x,act)
   for name in (('relu_scale','relu_uncoupled') if act=='relu' else ('tanh_sign','tanh_uncoupled')):
    path=folder/f'{condition}_{seed}_{name}.zip';raw=path.read_bytes();arr=run.load(path)
    row=next(z for z in d['summaries'] if z['activation']==act and z['method']==name)
    assert len(raw)==row['serialized_bytes'] and hashlib.sha256(raw).hexdigest()==row['payload_sha256']
    pred=run.eval_arrays(arr,x,act);diff=float((pred-ref).abs().max())
    assert abs(diff-row['max_abs_output_difference'])<1e-12
    if name in ('relu_scale','tanh_sign'):
     codes=run.symmetry_codes(act)
     fp=run.transform(model,name,codes)
     with torch.no_grad():fpout=run.forward(x,fp,act);baseout=run.forward(x,model,act)
     fp32=float((fpout-baseout).abs().max());max_fp32=max(max_fp32,fp32)
     assert abs(fp32-row['fp32_symmetry_max_abs_difference'])<1e-12
     assert fp32<=1e-5
    checked+=1
print(json.dumps({'payload_and_metric_rows_checked':checked,'max_fp32_symmetry_difference':max_fp32}))
