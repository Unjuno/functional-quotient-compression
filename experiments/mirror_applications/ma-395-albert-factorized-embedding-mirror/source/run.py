"""Held-out domain screen for Mirror inside a factorized embedding bottleneck."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from model import CLASSES,DIM,DOMAINS,LATENT,METHODS,VOCAB,FactorizedDomainBank,compute_proxy,from_payload

torch.set_num_threads(1);UPDATES=2000;BATCH=256;LR=0.02


def rotate(z:torch.Tensor,angles:torch.Tensor)->torch.Tensor:
    x=z.unsqueeze(0).expand(angles.shape[0],-1,-1);c=torch.cos(angles)[:,None,:];s=torch.sin(angles)[:,None,:]
    y=torch.empty_like(x);y[:,:,0::2]=c*x[:,:,0::2]-s*x[:,:,1::2];y[:,:,1::2]=s*x[:,:,0::2]+c*x[:,:,1::2]
    return y


def make_world(seed:int)->dict[str,object]:
    g=torch.Generator().manual_seed(seed);base=0.35*torch.randn(VOCAB,LATENT,generator=g)
    projection=torch.randn(LATENT,DIM,generator=g)/LATENT**0.5
    angles=0.4*torch.randn(DOMAINS,LATENT//2,generator=g);angles[0].zero_()
    teacher=rotate(base,angles)@projection
    decoder=0.5*torch.randn(DIM,CLASSES,generator=g)/DIM**0.5
    labels=(teacher@decoder).argmax(-1)
    sg=torch.Generator().manual_seed(seed+6001);held=torch.rand(DOMAINS,VOCAB,generator=sg)<0.2
    for t in range(VOCAB):
        if bool(held[:,t].all()):held[int(torch.randint(DOMAINS,(1,),generator=sg)),t]=False
    for d in range(DOMAINS):
        if bool(held[d].all()):held[d,int(torch.randint(VOCAB,(1,),generator=sg))]=False
    pairs=torch.nonzero(~held,as_tuple=False)
    return {'seed':seed,'base':base,'projection':projection,'angles':angles,'teacher':teacher,'decoder':decoder,
            'labels':labels,'heldout':held,'train_pairs':pairs,'train_count':int(pairs.shape[0]),
            'heldout_count':int(held.sum())}


def nrmse(x:torch.Tensor,y:torch.Tensor)->float:return float(torch.sqrt((x-y).square().mean()/y.square().mean()))


def score(bank:FactorizedDomainBank,w:dict[str,object])->dict[str,float]:
    ds=torch.arange(DOMAINS).repeat_interleave(VOCAB);ts=torch.arange(VOCAB).repeat(DOMAINS)
    target=w['teacher'].reshape(-1,DIM);labels=w['labels'].reshape(-1);held=w['heldout'].reshape(-1)
    with torch.no_grad():
        emb=bank(ts,ds);logits=emb@bank.decoder;out={}
        for tag,mask in [('observed',~held),('heldout',held)]:
            out[f'{tag}_embedding_nrmse']=nrmse(emb[mask],target[mask])
            out[f'{tag}_decoder_nll']=float(F.cross_entropy(logits[mask],labels[mask]))
            out[f'{tag}_decoder_top1_accuracy']=float((logits[mask].argmax(1)==labels[mask]).float().mean())
    return out


def fit(method:str,seed:int,w:dict[str,object])->tuple[FactorizedDomainBank,dict[str,float],float,int]:
    oracle=w['teacher'] if method=='oracle' else None
    bank=FactorizedDomainBank(method,w['base'],w['projection'],w['decoder'],oracle)
    params=list(bank.parameters());updates=UPDATES if params else 0;start=time.perf_counter()
    if params:
        opt=torch.optim.Adam(params,lr=LR);gen=torch.Generator().manual_seed(seed+8001);pairs=w['train_pairs']
        for _ in range(UPDATES):
            rows=pairs[torch.randint(pairs.shape[0],(BATCH,),generator=gen)];d=rows[:,0];t=rows[:,1]
            loss=(bank(t,d)-w['teacher'][d,t]).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    elapsed=time.perf_counter()-start;bank.eval()
    return bank,score(bank,w),elapsed,updates


def save_payload(path:Path,bank:FactorizedDomainBank)->tuple[int,str]:
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**bank.payload_arrays())
    raw=path.read_bytes();return len(raw),hashlib.sha256(raw).hexdigest()


def load_payload(path:Path)->FactorizedDomainBank:
    with np.load(path,allow_pickle=False) as z:return from_payload({k:z[k] for k in z.files})


def run(seed:int,condition:str,outdir:Path)->dict[str,object]:
    w=make_world(seed);rows=[]
    for method in METHODS:
        t0=time.perf_counter();bank,metrics,train_s,updates=fit(method,seed,w)
        path=outdir/f'{condition}_{seed}_{method}.npz';size,digest=save_payload(path,bank)
        bank=load_payload(path);metrics=score(bank,w)
        ds=torch.arange(DOMAINS).repeat_interleave(VOCAB);ts=torch.arange(VOCAB).repeat(DOMAINS)
        ti=time.perf_counter()
        with torch.no_grad():_=bank(ts,ds)@bank.decoder
        inf=time.perf_counter()-ti
        rows.append({'method':method,'payload_path':path.name,'serialized_bytes':size,'payload_sha256':digest,
             **metrics,**compute_proxy(method),'optimizer_updates':updates,'observed_pairs_seen':updates*BATCH,
             'training_wall_s':train_s,'inference_pairs_per_s':DOMAINS*VOCAB/max(inf,1e-9),'total_wall_s':time.perf_counter()-t0})
    return {'experiment_id':'MA-395','condition':condition,'seed':seed,'train_pair_count':w['train_count'],
            'heldout_pair_count':w['heldout_count'],'updates_per_trainable_method':UPDATES,'batch_size':BATCH,
            'learning_rate':LR,'results':rows}


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--condition',choices=('development','fresh'),required=True)
    p.add_argument('--outdir',type=Path,required=True);p.add_argument('--json',type=Path,required=True);a=p.parse_args()
    res=run(a.seed,a.condition,a.outdir);a.json.parent.mkdir(parents=True,exist_ok=True);a.json.write_text(json.dumps(res,indent=2)+'\n')
    print(json.dumps({'seed':a.seed,'condition':a.condition,'train_pairs':res['train_pair_count'],'heldout_pairs':res['heldout_pair_count'],
      'results':[{k:r[k] for k in ('method','serialized_bytes','heldout_embedding_nrmse','heldout_decoder_nll','heldout_decoder_top1_accuracy')} for r in res['results']]},indent=2))


if __name__=='__main__':main()
