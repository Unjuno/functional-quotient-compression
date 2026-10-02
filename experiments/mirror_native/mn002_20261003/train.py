# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Train one predeclared MN002 run; audit is deliberately a separate command."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from model import TinyLM
from task import make_table,dataset,check_splits
from utils import dump,evaluate,environment
ROOT=Path(__file__).resolve().parent

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def run(name: str, seed: int, output: Path, protocol_path: Path=ROOT/'protocol.json') -> dict:
    protocol=json.loads(protocol_path.read_text())
    variant=next((v for v in protocol['variants'] if v['name']==name),None)
    if variant is None or seed not in protocol['seeds']:
        raise ValueError('variant and seed must be predeclared')
    output.mkdir(parents=True,exist_ok=True)
    target=output/f'{name}_seed{seed}'
    if target.with_suffix('.json').exists() or target.with_suffix('.pt').exists():
        raise FileExistsError('refusing to overwrite an existing run')
    torch.set_num_threads(protocol['threads'])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    cfg={k:protocol[k] for k in ('d','layers','heads','max_length')}
    cfg.update({k:variant[k] for k in ('kind','hidden','states')})
    cfg['vocab']=protocol['symbols']+protocol['rules']+1
    model=TinyLM(**cfg)
    initial={k:v.detach().clone() for k,v in model.named_parameters()}
    table=make_table(protocol['rules'],protocol['symbols'],protocol['table_seed'])
    train=dataset(protocol['train_sequences'],protocol['train_seed'],table,protocol['records'])
    dev=dataset(protocol['development_sequences'],protocol['development_seed'],table,protocol['records'])
    split_check=check_splits(train,dev)
    pair_ids=(train[0][:,1::3]-protocol['symbols'])*protocol['symbols']+train[0][:,2::3]
    if pair_ids.unique().numel()!=table.numel(): raise RuntimeError('training lacks a rule-symbol pair')
    before=evaluate(model.eval(),dev)
    opt=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],betas=tuple(protocol['betas']),weight_decay=protocol['weight_decay'])
    batch_rng=torch.Generator().manual_seed(45000+seed)
    trace=[]; start=time.perf_counter()
    for step in range(1,protocol['steps']+1):
        model.train()
        idx=torch.randint(len(train[0]),(protocol['batch'],),generator=batch_rng)
        opt.zero_grad(set_to_none=True)
        logits,_,_=model(train[0][idx])
        loss=F.cross_entropy(logits.flatten(0,1),train[1][idx].flatten(),ignore_index=-100)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
        loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['clip_norm'])
        if not torch.isfinite(grad_norm): raise RuntimeError('nonfinite gradient')
        opt.step()
        if step%100==0 or step==protocol['steps']:
            dv=evaluate(model.eval(),dev)
            row={'step':step,'train_nll':loss.item(),'dev_nll':dv['nll'],'dev_accuracy':dv['accuracy']}
            trace.append(row)
            print(json.dumps({'name':name,'seed':seed,**row}),flush=True)
    elapsed=time.perf_counter()-start
    model.eval()
    ckpt=target.with_suffix('.pt')
    torch.save({'config':model.config,'state_dict':model.state_dict(),'steps':protocol['steps'],'seed':seed,'protocol_sha256':sha(protocol_path),'variant':name},ckpt)
    saved=torch.load(ckpt,weights_only=True,map_location='cpu')
    restored=TinyLM(**saved['config']).eval();restored.load_state_dict(saved['state_dict'])
    with torch.no_grad(): reload_error=(restored(dev[0][:8])[0]-model(dev[0][:8])[0]).abs().max().item()
    assert reload_error==0.0
    dv=evaluate(restored,dev)
    fixed=[r['mean_prob_all_tokens'] for r in dv.get('routing',[])]
    d,h,s=cfg['d'],cfg['hidden'],cfg['states']
    ff_macs=2*d*h
    if cfg['kind']=='mirror': ff_macs+=d*s+s*h
    if cfg['kind']=='direct_gate': ff_macs+=d*h
    if cfg['kind']=='full_moe': ff_macs=2*d*h*s+d*s
    result={'experiment':protocol['experiment'],'name':name,'seed':seed,'config':cfg,'steps':protocol['steps'],'batch':protocol['batch'],
            'parameters':sum(p.numel() for p in model.parameters()),
            'parameter_bytes_fp32':sum(p.numel()*p.element_size() for p in model.parameters()),
            'checkpoint_bytes':ckpt.stat().st_size,'checkpoint_sha256':sha(ckpt),'protocol_sha256':sha(protocol_path),
            'ff_linear_macs_per_token_all_layers':cfg['layers']*ff_macs,
            'linear_macs_note':'Linear MACs only; no claim of matched total runtime, softmax/nonlinearities/dispatch cost.',
            'initial_dev':{k:before[k] for k in ('nll','accuracy')},'trace':trace,
            'development':dv,'fixed_development_routing':fixed,
            'parameter_update_norm':{k:(v.detach()-initial[k]).norm().item() for k,v in model.named_parameters()},
            'training_seconds_including_dev':elapsed,'reload_error_max':reload_error,
            'splits':split_check,'all_rule_symbol_pairs_seen':True,'environment':environment()}
    dump(target.with_suffix('.json'),result)
    print('TRAIN_DONE',name,seed,'seconds',elapsed,'params',result['parameters'],flush=True)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name',required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--output',type=Path,default=ROOT/'results/train')
    ap.add_argument('--protocol',type=Path,default=ROOT/'protocol.json')
    a=ap.parse_args();run(a.name,a.seed,a.output,a.protocol)
