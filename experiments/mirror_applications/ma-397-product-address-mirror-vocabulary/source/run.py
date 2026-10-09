"""Synthetic product-address collision-resolution experiment."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from model import BUCKETS,CLASSES,DIM,METHODS,VOCAB,ProductBank,compute_proxy,product_indices,from_payload

torch.set_num_threads(1);UPDATES=4000;BATCH=256;LR=0.01


def make_world(seed:int)->dict[str,object]:
    g=torch.Generator().manual_seed(seed)
    t0=0.4*torch.randn(BUCKETS,DIM,generator=g);t1=0.4*torch.randn(BUCKETS,DIM,generator=g)
    angles=2*torch.pi*torch.rand(VOCAB,generator=g);tokens=torch.arange(VOCAB);a,b=product_indices(tokens)
    teacher=torch.cos(angles)[:,None]*t0[a]+torch.sin(angles)[:,None]*t1[b]
    decoder=0.5*torch.randn(DIM,CLASSES,generator=g)/DIM**0.5
    labels=(teacher@decoder).argmax(1)
    key=a*BUCKETS+b;counts=torch.bincount(key,minlength=BUCKETS*BUCKETS)
    return {'seed':seed,'table0':t0,'table1':t1,'angles':angles,'teacher':teacher,'decoder':decoder,'labels':labels,
            'unique_addresses':int((counts>0).sum()),'address_occupancy_min':int(counts[counts>0].min()),
            'address_occupancy_max':int(counts.max()),'address_collision_count':int((counts>1).sum())}


def nrmse(x:torch.Tensor,y:torch.Tensor)->float:return float(torch.sqrt((x-y).square().mean()/y.square().mean()))


def score(bank:ProductBank,w:dict[str,object])->dict[str,float]:
    ids=torch.arange(VOCAB)
    with torch.no_grad():
        out=bank(ids);logits=out@bank.decoder;pair_pred=out[:VOCAB//2]-out[VOCAB//2:]
        pair_true=w['teacher'][:VOCAB//2]-w['teacher'][VOCAB//2:]
        return {'embedding_nrmse':nrmse(out,w['teacher']),'collision_pair_separation_nrmse':nrmse(pair_pred,pair_true),
            'decoder_nll':float(F.cross_entropy(logits,w['labels'])),'decoder_top1_accuracy':float((logits.argmax(1)==w['labels']).float().mean()),
            'mean_predicted_collision_separation':float(pair_pred.norm(dim=1).mean()),
            'mean_teacher_collision_separation':float(pair_true.norm(dim=1).mean())}


def fit(method:str,seed:int,w:dict[str,object])->tuple[ProductBank,dict[str,float],float,int]:
    bank=ProductBank(method,seed+1201,w['table0'],w['table1'],w['decoder']);params=list(bank.parameters())
    updates=UPDATES if params else 0;start=time.perf_counter();target=w['teacher']
    if params:
        opt=torch.optim.Adam(params,lr=LR);g=torch.Generator().manual_seed(seed+2603)
        for _ in range(UPDATES):
            ids=torch.randint(VOCAB,(BATCH,),generator=g);pred=bank(ids);loss=(pred-target[ids]).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    elapsed=time.perf_counter()-start;bank.eval();return bank,score(bank,w),elapsed,updates


def save_payload(path:Path,bank:ProductBank)->tuple[int,str]:
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**bank.payload_arrays())
    raw=path.read_bytes();return len(raw),hashlib.sha256(raw).hexdigest()


def load_payload(path:Path)->ProductBank:
    with np.load(path,allow_pickle=False) as z:return from_payload({k:z[k] for k in z.files})


def run(seed:int,condition:str,outdir:Path)->dict[str,object]:
    w=make_world(seed);rows=[]
    for method in METHODS:
        t0=time.perf_counter();bank,metrics,train_s,updates=fit(method,seed,w)
        path=outdir/f'{condition}_{seed}_{method}.npz';size,digest=save_payload(path,bank)
        bank=load_payload(path);metrics=score(bank,w);ids=torch.arange(VOCAB);start=time.perf_counter()
        with torch.no_grad():_=bank(ids)@bank.decoder
        infer=time.perf_counter()-start
        rows.append({'method':method,'payload_path':path.name,'serialized_bytes':size,'payload_sha256':digest,**metrics,
            **compute_proxy(method),'optimizer_updates':updates,'training_tokens_seen':updates*BATCH,
            'training_wall_s':train_s,'inference_tokens_per_s':VOCAB/max(infer,1e-9),'total_wall_s':time.perf_counter()-t0})
    return {'experiment_id':'MA-397','condition':condition,'seed':seed,'vocabulary_size':VOCAB,
        'unique_addresses':w['unique_addresses'],'address_collision_count':w['address_collision_count'],
        'address_occupancy_min':w['address_occupancy_min'],'address_occupancy_max':w['address_occupancy_max'],
        'updates':UPDATES,'batch_size':BATCH,'learning_rate':LR,'results':rows}


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--condition',choices=('development','fresh'),required=True)
    p.add_argument('--outdir',type=Path,required=True);p.add_argument('--json',type=Path,required=True);a=p.parse_args()
    d=run(a.seed,a.condition,a.outdir);a.json.parent.mkdir(parents=True,exist_ok=True);a.json.write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps({'seed':a.seed,'condition':a.condition,'addresses':d['unique_addresses'],'collisions':d['address_collision_count'],
      'results':[{k:r[k] for k in ('method','serialized_bytes','embedding_nrmse','collision_pair_separation_nrmse','decoder_top1_accuracy')} for r in d['results']]},indent=2))


if __name__=='__main__':main()
