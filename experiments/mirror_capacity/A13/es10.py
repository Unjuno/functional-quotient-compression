# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A13d 10× ES restarts for frozen tentative Mirror scaffold only."""
from pathlib import Path
import argparse,json,hashlib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from a13 import es_optimize,make_world,deserialize,serialize,make_combined_model,POP,GENERATIONS,score_support

ROOT=Path(__file__).resolve().parent

def sha(b):return hashlib.sha256(b).hexdigest()

def one(args):
    source,entry,out=args
    torch.set_num_threads(1)
    p=Path(source)/entry['path']
    raw=p.read_bytes()
    if sha(raw)!=entry['sha256']:raise RuntimeError('parent changed')
    m=deserialize(raw)
    world=make_world(entry['world'],entry['family'])
    c=[];restarts=[];t0=time.perf_counter()
    for task in range(8):
        start=m.new_codes[task].detach().numpy().copy()
        original=40000011+entry['world']*1973+entry['seed']*17+task*109
        best=start.copy();best_loss=score_support(m,world['support_x'][task],world['support_y'][task],start)
        best_rep=-1
        for rep in range(10):
            es=es_optimize(m,world['support_x'][task],world['support_y'][task],start,original+7919*rep,pop=POP,generations=GENERATIONS)
            if es['best_support_mse']<best_loss:
                best=es['code'].copy();best_loss=float(es['best_support_mse']);best_rep=rep
        c.append(best)
        restarts.append(dict(task=task,best_replica=best_rep,best_support_mse=best_loss))
    model=make_combined_model(m,np.array(c))
    data=serialize(model)
    path=Path(out)/'models'/f"{entry['family']}_w{entry['world']}_s{entry['seed']}_{m.method}_ES10.a11w"
    path.parent.mkdir(exist_ok=True)
    with path.open('xb') as f:f.write(data)
    return {'path':str(path.relative_to(out)),'family':entry['family'],'world':entry['world'],'seed':entry['seed'],
            'method':entry['method'],'algorithm':'es10','source_experiment':source.name,
            'parent_sha256':entry['sha256'],'world_fingerprint':world['fingerprint'],
            'sha256':sha(data),'stored_bytes':len(data),
            'support_mse_mean':float(np.mean([r['best_support_mse'] for r in restarts])),
            'elapsed_s':time.perf_counter()-t0,'best_restarts':[z['best_replica'] for z in restarts]}

def run(out:Path,main:Path,replication:Path):
    out=out.resolve();out.mkdir(exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('refusing overwrite')
    sources=[main.resolve(),replication.resolve()]
    cfg={'experiment':'A13D_POSTHOC_10X_ES','source_ledger_hashes':{s.name:sha((s/'train.lock.json').read_bytes()) for s in sources},
         'protocol_sha256':sha((ROOT/'PROTOCOL_A13D.md').read_bytes()),'script_sha256':sha((ROOT/'es10.py').read_bytes()),
         'methods':['mirror'],'families':['tent'],'worlds':[1200,1201,1202,1300,1301,1302],
         'restarts':10,'evaluations_per_restart_task':POP*GENERATIONS,'selection':'only 32 support labels; no test data'}
    raw_cfg=json.dumps(cfg,sort_keys=True,indent=2).encode();(out/'config.json').write_bytes(raw_cfg)
    entries=[]
    for s in sources:
        lock=json.loads((s/'train.lock.json').read_bytes())
        entries.extend([(s,p) for p in lock['parents'] if p.get('family')=='tent' and p.get('method')=='mirror'])
        # Main A13 parent rows lack a family/world/seed field; join via the checkpoint child ledger.
        if s.name=='main':
            rows=lock['models']; lookup={}
            for a in rows:lookup[a['parent_sha256']]={k:a[k] for k in ('family','world','seed','method')}
            entries.extend([(s,{**p,**lookup[p['sha256']]}) for p in lock['parents'] if p['sha256'] in lookup and lookup[p['sha256']]['family']=='tent' and lookup[p['sha256']]['method']=='mirror'])
    if len(entries)!=12:raise RuntimeError('wrong frozen Mirror parent count')
    outrows=[];tic=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(one,(s,p,str(out))) for s,p in entries]):
            r=f.result();outrows.append(r)
            print('ES10',len(outrows),'/',len(entries),r['world'],r['seed'],round(r['elapsed_s'],2),flush=True)
    outrows.sort(key=lambda x:(x['world'],x['seed']))
    lock=dict(config_sha256=sha(raw_cfg),models=outrows,elapsed_s=time.perf_counter()-tic)
    (out/'train.lock.json').write_text(json.dumps(lock,sort_keys=True,indent=2))
    print('ES10_LOCKED',len(outrows),'models',lock['elapsed_s'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'es10');ap.add_argument('--main',type=Path,default=ROOT/'main');ap.add_argument('--replication',type=Path,default=ROOT/'replication');a=ap.parse_args();run(a.out,a.main,a.replication)
