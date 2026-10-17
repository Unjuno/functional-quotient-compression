"""Frozen, sequential CPU feasibility runs. Never tune or truncate fresh runs."""
import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from model import Transformer, MODES, batch_data, world_encoding, pack, unpack

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(model,task,perm,seed,n=1024):
    gen=torch.Generator().manual_seed(seed)
    total=correct=0
    model.eval()
    with torch.no_grad():
        for start in range(0,n,64):
            x,y=batch_data(task,min(64,n-start),gen,perm)
            z=model(x)
            total+=F.cross_entropy(z,y,reduction='sum').item()
            correct+=int((z.argmax(-1)==y).sum())
    return total/n,correct/n


def forward_macs(model):
    c=model.cfg; l=c.length; d=c.d
    return c.layers*(4*l*d*d+2*l*d*c.ff+2*l*l*d)+d*c.vocab


def latency(model,batch):
    x=torch.zeros(batch,16,dtype=torch.long)
    values=[]
    with torch.no_grad():
        for _ in range(5): model(x)
        for _ in range(25):
            start=time.perf_counter_ns(); model(x)
            values.append((time.perf_counter_ns()-start)/1e6)
    return np.percentile(values,[10,50,90]).tolist()


def run_one(mode,task,world,seed,steps,eval_n,phase):
    torch.manual_seed(world*13+seed)
    model=Transformer(mode)
    initial_codes=model.codes.detach().clone() if hasattr(model,'codes') else None
    perm=world_encoding(world)
    stream=world*101+seed*1009+(0 if task=='copy' else 100000003)
    gen=torch.Generator().manual_seed(stream)
    eval_seed=stream+1000000007
    nll0,acc0=evaluate(model,task,perm,eval_seed,eval_n)
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=0.)
    trace=[]; start=time.perf_counter()
    model.train()
    for step in range(1,steps+1):
        x,y=batch_data(task,32,gen,perm)
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x),y)
        if not torch.isfinite(loss):
            raise RuntimeError(f'nonfinite loss: {mode}/{task}/{world}/{seed}/{step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        opt.step()
        if step==1 or step%100==0 or step==steps:
            trace.append({'step':step,'batch_loss':float(loss.detach())})
    seconds=time.perf_counter()-start
    nll,acc=evaluate(model,task,perm,eval_seed,eval_n)
    blob=pack(model)
    name=f'{phase}_{mode}_{task}_{world}_{seed}'
    path=ROOT/'checkpoints'/f'{name}.pt'
    if path.exists(): raise FileExistsError(path)
    path.write_bytes(blob)
    fixture,_=batch_data(task,8,torch.Generator().manual_seed(12),perm)
    with torch.no_grad():
        a=model(fixture)
        restored=unpack(blob)
        roundtrip=bool(torch.equal(a,restored(fixture)))
        altered=fixture.clone(); altered[:,9:]=(altered[:,9:]+1)%64
        prefix_error=float((model(fixture,True)[:,:9]-model(altered,True)[:,:9]).abs().max())
    timing1=latency(model,1); timing32=latency(model,32)
    physical=sum(p.numel() for p in model.parameters())
    stored=sum(v.numel() for v in model.state_dict().values())
    row={'phase':phase,'model':mode,'task':task,'world':world,'seed':seed,
         'updates':steps,'batch':32,'length':16,'examples':steps*32,'tokens':steps*32*16,
         'initial_nll':nll0,'initial_acc':acc0,'nll':nll,'accuracy':acc,
         'learned_scalars':physical,'frozen_stored_scalars':stored-physical,
         'logical_weight_elements':sum(a*b for a,b in model.shapes),
         'raw_state_bytes':stored*4,'payload_bytes':len(blob),
         'forward_dense_macs_per_example':forward_macs(model),
         'train_dense_macs_proxy_3x':3*steps*32*forward_macs(model),
         'train_s':seconds,'infer_b1_ms_p10':timing1[0],'infer_b1_ms_median':timing1[1],
         'infer_b1_ms_p90':timing1[2],'infer_b32_ms_p10':timing32[0],
         'infer_b32_ms_median':timing32[1],'infer_b32_ms_p90':timing32[2],
         'causal_prefix_max_abs':prefix_error,'roundtrip_exact':roundtrip,
         'checkpoint_sha256':hashlib.sha256(blob).hexdigest(),
         'code_max_change':float((model.codes.detach()-initial_codes).abs().max()) if initial_codes is not None else 0.,
         'zero_codes_nll':None,'zero_codes_acc':None}
    if mode=='mirror':
        saved=model.codes.detach().clone()
        with torch.no_grad(): model.codes.zero_()
        row['zero_codes_nll'],row['zero_codes_acc']=evaluate(model,task,perm,eval_seed,eval_n)
        with torch.no_grad(): model.codes.copy_(saved)
    (ROOT/'results'/f'{name}.trace.json').write_text(json.dumps(trace,indent=2))
    if not roundtrip or prefix_error>1e-6:
        raise AssertionError(f'causality or reload failed: {row}')
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['smoke','fresh'],required=True)
    p.add_argument('--models',default=','.join(MODES))
    p.add_argument('--tasks',default='copy,retrieve')
    p.add_argument('--smoke-steps',type=int,default=20)
    a=p.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    worlds=[618230] if a.phase=='smoke' else [618231,618232,618233]
    seeds=[0] if a.phase=='smoke' else [0,1]
    steps=a.smoke_steps if a.phase=='smoke' else 400
    if a.phase=='fresh' and (a.models!=','.join(MODES) or a.tasks!='copy,retrieve'):
        raise ValueError('fresh grid must be complete; no post-selection subsets')
    frozen=json.loads((ROOT/'FREEZE.json').read_text())
    for name,value in frozen['sha256'].items():
        if sha(ROOT/name)!=value: raise ValueError(f'source changed after freeze: {name}')
    result=ROOT/'results'/f'{a.phase}.csv'
    if result.exists(): raise FileExistsError(result)
    hw={k:v for k,v in [l.split(':',1) for l in Path('/proc/cpuinfo').read_text().splitlines() if ':' in l]}
    meta={'phase':a.phase,'platform':platform.platform(),'torch':torch.__version__,
          'cpu_model':hw.get('model name\t','unknown').strip(),'logical_cpus':os.cpu_count(),
          'torch_threads':1,'dtype':'float32','device':'cpu','clock':'not pinned; virtualized CPU',
          'batch':32,'length':16,'updates':steps,'worlds':worlds,'seeds':seeds,
          'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'source_sha256':sha(ROOT/'source/model.py')}
    (ROOT/'results'/f'{a.phase}.meta.json').write_text(json.dumps(meta,indent=2))
    rows=[]
    for task in a.tasks.split(','):
        for world in worlds:
            for seed in seeds:
                # Rotate execution order deterministically; runs remain sequential.
                models=a.models.split(','); shift=(world+seed)%len(models)
                for mode in models[shift:]+models[:shift]:
                    row=run_one(mode,task,world,seed,steps,256 if a.phase=='smoke' else 1024,a.phase)
                    rows.append(row)
                    with result.open('w',newline='') as out:
                        w=csv.DictWriter(out,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
                    print(f'{len(rows):02d} {mode:12} {task:8} w={world} s={seed} '
                          f'NLL {row["initial_nll"]:.3f}->{row["nll"]:.3f} '
                          f'acc={row["accuracy"]:.3f} time={row["train_s"]:.1f}s',flush=True)
    (ROOT/'results'/f'{a.phase}.complete.json').write_text(json.dumps({'rows':len(rows),'csv_sha256':sha(result)},indent=2))

if __name__=='__main__': main()
