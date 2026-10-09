"""Synthetic token x domain held-out generalization screen for MA-392."""
from __future__ import annotations

import argparse,hashlib,json,time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import CLASSES,DIM,DOMAINS,METHODS,VOCAB,DomainBank,compute_proxy,from_payload

torch.set_num_threads(1)
UPDATES=2000
BATCH=256
LR=0.02


def apply_rotation(table:torch.Tensor,angles:torch.Tensor)->torch.Tensor:
    # angles [domain, plane], table [token, feature]
    x=table.unsqueeze(0).expand(angles.shape[0],-1,-1)
    c=torch.cos(angles)[:,None,:]; s=torch.sin(angles)[:,None,:]
    y=torch.empty_like(x)
    y[:,:,0::2]=c*x[:,:,0::2]-s*x[:,:,1::2]
    y[:,:,1::2]=s*x[:,:,0::2]+c*x[:,:,1::2]
    return y


def make_world(seed:int)->dict[str,object]:
    g=torch.Generator().manual_seed(seed)
    base=0.35*torch.randn(VOCAB,DIM,generator=g)
    angles=0.45*torch.randn(DOMAINS,DIM//2,generator=g); angles[0].zero_()
    teacher=apply_rotation(base,angles)
    decoder=0.5*torch.randn(DIM,CLASSES,generator=g)/DIM**0.5
    labels=(teacher@decoder).argmax(dim=-1)
    split_g=torch.Generator().manual_seed(seed+4001)
    heldout=torch.rand(DOMAINS,VOCAB,generator=split_g)<0.20
    # Keep the split identifiable: every token has at least one observed domain,
    # and every domain has observed tokens.
    for token in range(VOCAB):
        if bool(heldout[:,token].all()): heldout[int(torch.randint(DOMAINS,(1,),generator=split_g)),token]=False
    for domain in range(DOMAINS):
        if bool(heldout[domain].all()): heldout[domain,int(torch.randint(VOCAB,(1,),generator=split_g))]=False
    train_pairs=torch.nonzero(~heldout,as_tuple=False) # [domain, token]
    return {'seed':seed,'base':base,'angles':angles,'teacher':teacher,'decoder':decoder,
            'labels':labels,'heldout_mask':heldout,'train_pairs':train_pairs,
            'train_pair_count':int(train_pairs.shape[0]),'heldout_pair_count':int(heldout.sum())}


def nrmse(pred:torch.Tensor,target:torch.Tensor)->float:
    return float(torch.sqrt(torch.mean((pred-target)**2)/torch.mean(target**2)))


def score(bank:DomainBank,world:dict[str,object])->dict[str,float]:
    domains=torch.arange(DOMAINS).repeat_interleave(VOCAB)
    tokens=torch.arange(VOCAB).repeat(DOMAINS)
    flat_teacher=world['teacher'].reshape(DOMAINS*VOCAB,DIM)
    flat_labels=world['labels'].reshape(-1)
    heldout=world['heldout_mask'].reshape(-1)
    with torch.no_grad():
        emb=bank(tokens,domains); logits=emb@bank.decoder
        result={}
        for name,mask in [('observed',~heldout),('heldout',heldout)]:
            e=emb[mask]; target=flat_teacher[mask]; out=logits[mask]; y=flat_labels[mask]
            result[f'{name}_embedding_nrmse']=nrmse(e,target)
            result[f'{name}_decoder_nll']=float(F.cross_entropy(out,y))
            result[f'{name}_decoder_top1_accuracy']=float((out.argmax(1)==y).float().mean())
    return result


def fit(method:str,seed:int,world:dict[str,object])->tuple[DomainBank,dict[str,float],float,int]:
    oracle=world['teacher'] if method=='oracle' else None
    bank=DomainBank(method,world['base'],world['decoder'],oracle)
    params=list(bank.parameters()); updates=UPDATES if params else 0
    start=time.perf_counter()
    if params:
        opt=torch.optim.Adam(params,lr=LR)
        gen=torch.Generator().manual_seed(seed+5003)
        train=world['train_pairs']
        for _ in range(UPDATES):
            idx=torch.randint(train.shape[0],(BATCH,),generator=gen)
            pairs=train[idx]; d=pairs[:,0]; t=pairs[:,1]
            pred=bank(t,d); target=world['teacher'][d,t]
            loss=torch.mean((pred-target)**2)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    elapsed=time.perf_counter()-start
    bank.eval()
    return bank,score(bank,world),elapsed,updates


def save_payload(path:Path,bank:DomainBank)->tuple[int,str]:
    path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,**bank.payload_arrays())
    raw=path.read_bytes(); return len(raw),hashlib.sha256(raw).hexdigest()


def load_payload(path:Path)->DomainBank:
    with np.load(path,allow_pickle=False) as z: payload={key:z[key] for key in z.files}
    return from_payload(payload)


def run(seed:int,condition:str,outdir:Path)->dict[str,object]:
    world=make_world(seed); rows=[]
    for method in METHODS:
        t0=time.perf_counter(); bank,metrics,train_s,updates=fit(method,seed,world)
        path=outdir/f'{condition}_{seed}_{method}.npz'; size,digest=save_payload(path,bank)
        bank=load_payload(path); metrics=score(bank,world)
        ds=torch.arange(DOMAINS).repeat_interleave(VOCAB); ts=torch.arange(VOCAB).repeat(DOMAINS)
        t1=time.perf_counter()
        with torch.no_grad(): _=bank(ts,ds)@bank.decoder
        infer_s=time.perf_counter()-t1
        rows.append({'method':method,'payload_path':path.name,'serialized_bytes':size,'payload_sha256':digest,
                     **metrics,**compute_proxy(method),'optimizer_updates':updates,
                     'observed_pairs_seen':updates*BATCH,'training_wall_s':train_s,
                     'inference_pairs_per_s':DOMAINS*VOCAB/max(infer_s,1e-9),
                     'total_wall_s':time.perf_counter()-t0})
    return {'experiment_id':'MA-392','condition':condition,'seed':seed,'vocabulary_size':VOCAB,
            'domain_count':DOMAINS,'embedding_dim':DIM,'train_pair_count':world['train_pair_count'],
            'heldout_pair_count':world['heldout_pair_count'],'updates_per_trainable_method':UPDATES,
            'batch_size':BATCH,'learning_rate':LR,'results':rows}


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--condition',choices=('development','fresh'),required=True)
    p.add_argument('--outdir',type=Path,required=True);p.add_argument('--json',type=Path,required=True)
    a=p.parse_args();result=run(a.seed,a.condition,a.outdir)
    a.json.parent.mkdir(parents=True,exist_ok=True);a.json.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'seed':a.seed,'condition':a.condition,'train_pairs':result['train_pair_count'],
        'heldout_pairs':result['heldout_pair_count'],'results':[{k:r[k] for k in ('method','serialized_bytes','heldout_embedding_nrmse','heldout_decoder_nll','heldout_decoder_top1_accuracy','optimizer_updates')} for r in result['results']]},indent=2))


if __name__=='__main__':main()
