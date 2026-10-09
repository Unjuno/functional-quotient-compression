"""Aligned frequency-banded embedding screen for MA-393."""
from __future__ import annotations

import argparse,hashlib,json,time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import BAND_MASSES,BAND_SIZES,BAND_WIDTHS,CLASSES,DIM,METHODS,VOCAB,AdaptiveBank,compute_proxy,projection,rotate_first_pair,from_payload

torch.set_num_threads(1)
UPDATES=2000;BATCH=256;LR=0.02


def make_world(seed:int)->dict[str,object]:
    g=torch.Generator().manual_seed(seed);basis_seed=seed+81001
    codes=[];angles=[];targets=[];bands=[]
    for b,(n,k) in enumerate(zip(BAND_SIZES,BAND_WIDTHS)):
        z=torch.randn(n,k,generator=g)
        theta=2*torch.pi*torch.rand(n,generator=g)
        raw=z@projection(basis_seed,b,k).T
        target=rotate_first_pair(raw,theta)
        codes.append(z);angles.append(theta);targets.append(target);bands.extend([b]*n)
    teacher=torch.cat(targets);theta_all=torch.cat(angles);band_ids=torch.tensor(bands)
    decoder=0.5*torch.randn(DIM,CLASSES,generator=g)/DIM**0.5
    labels=(teacher@decoder).argmax(dim=1)
    token_prob=torch.zeros(VOCAB)
    start=0
    for mass,n in zip(BAND_MASSES,BAND_SIZES):token_prob[start:start+n]=mass/n;start+=n
    return {'seed':seed,'basis_seed':basis_seed,'codes':codes,'angles':theta_all,'teacher':teacher,
            'decoder':decoder,'labels':labels,'band_ids':band_ids,'token_prob':token_prob}


def nrmse(pred:torch.Tensor,target:torch.Tensor,weights:torch.Tensor|None=None)->float:
    err=(pred-target).square().mean(dim=-1);den=target.square().mean(dim=-1)
    if weights is None:return float(torch.sqrt(err.mean()/den.mean()))
    return float(torch.sqrt((weights*err).sum()/ (weights*den).sum()))


def score(bank:AdaptiveBank,world:dict[str,object])->dict[str,float]:
    ids=torch.arange(VOCAB);weights=world['token_prob'];teacher=world['teacher'];labels=world['labels']
    with torch.no_grad():
        emb=bank(ids);logits=emb@bank.decoder;per_nll=F.cross_entropy(logits,labels,reduction='none')
        pred=logits.argmax(1);per_acc=(pred==labels).float()
        out={'uniform_embedding_nrmse':nrmse(emb,teacher),
             'weighted_embedding_nrmse':nrmse(emb,teacher,weights),
             'uniform_decoder_nll':float(per_nll.mean()),
             'weighted_decoder_nll':float((weights*per_nll).sum()),
             'uniform_decoder_top1_accuracy':float(per_acc.mean()),
             'weighted_decoder_top1_accuracy':float((weights*per_acc).sum())}
        for b,n in enumerate(BAND_SIZES):
            start=sum(BAND_SIZES[:b]);sl=slice(start,start+n)
            out[f'band_{b}_embedding_nrmse']=nrmse(emb[sl],teacher[sl])
            out[f'band_{b}_decoder_top1_accuracy']=float(per_acc[sl].mean())
    return out


def fit(method:str,seed:int,world:dict[str,object])->tuple[AdaptiveBank,dict[str,float],float]:
    bank=AdaptiveBank(method,seed+3001,world['basis_seed'],world['decoder'])
    opt=torch.optim.Adam(bank.parameters(),lr=LR);gen=torch.Generator().manual_seed(seed+9001)
    start=time.perf_counter();probs=world['token_prob'];target=world['teacher']
    for _ in range(UPDATES):
        ids=torch.multinomial(probs,BATCH,replacement=True,generator=gen)
        pred=bank(ids);loss=(pred-target[ids]).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    elapsed=time.perf_counter()-start;bank.eval()
    return bank,score(bank,world),elapsed


def save_payload(path:Path,bank:AdaptiveBank)->tuple[int,str]:
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**bank.payload_arrays())
    raw=path.read_bytes();return len(raw),hashlib.sha256(raw).hexdigest()


def load_payload(path:Path)->AdaptiveBank:
    with np.load(path,allow_pickle=False) as z:payload={key:z[key] for key in z.files}
    return from_payload(payload)


def run(seed:int,condition:str,outdir:Path)->dict[str,object]:
    world=make_world(seed);rows=[]
    for method in METHODS:
        t0=time.perf_counter();bank,metrics,train_s=fit(method,seed,world)
        path=outdir/f'{condition}_{seed}_{method}.npz';size,digest=save_payload(path,bank)
        bank=load_payload(path);metrics=score(bank,world)
        ids=torch.arange(VOCAB);t1=time.perf_counter()
        with torch.no_grad():_=bank(ids)@bank.decoder
        infer_s=time.perf_counter()-t1
        rows.append({'method':method,'payload_path':path.name,'serialized_bytes':size,'payload_sha256':digest,
            **metrics,**compute_proxy(method),'optimizer_updates':UPDATES,'training_examples_seen':UPDATES*BATCH,
            'training_wall_s':train_s,'inference_tokens_per_s':VOCAB/max(infer_s,1e-9),
            'total_wall_s':time.perf_counter()-t0})
    return {'experiment_id':'MA-393','condition':condition,'seed':seed,'vocabulary_size':VOCAB,
        'band_sizes':list(BAND_SIZES),'band_widths':list(BAND_WIDTHS),'band_sampling_masses':list(BAND_MASSES),
        'updates':UPDATES,'batch_size':BATCH,'learning_rate':LR,'results':rows}


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--condition',choices=('development','fresh'),required=True)
    p.add_argument('--outdir',type=Path,required=True);p.add_argument('--json',type=Path,required=True)
    a=p.parse_args();result=run(a.seed,a.condition,a.outdir)
    a.json.parent.mkdir(parents=True,exist_ok=True);a.json.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'seed':a.seed,'condition':a.condition,'results':[{k:r[k] for k in ('method','serialized_bytes','weighted_embedding_nrmse','band_2_embedding_nrmse','weighted_decoder_nll','weighted_decoder_top1_accuracy')} for r in result['results']]},indent=2))


if __name__=='__main__':main()
