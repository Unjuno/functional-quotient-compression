"""One-step exact speculative decoding proxy on nested MLP roles."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from model import INPUT,WIDTH,VOCAB,METHODS,SpecSystem,compute_proxy,speculative_distribution,from_payload

torch.set_num_threads(1);UPDATES=1200;BATCH=128;LR=0.01


def make_world(seed:int)->dict[str,object]:
    g=torch.Generator().manual_seed(seed)
    w1=torch.randn(INPUT,WIDTH,generator=g)/INPUT**0.5;b1=0.1*torch.randn(WIDTH,generator=g)
    w2=torch.randn(WIDTH,VOCAB,generator=g)/WIDTH**0.5;b2=0.1*torch.randn(VOCAB,generator=g)
    train=torch.randn(2048,INPUT,generator=g);evalx=torch.randn(2048,INPUT,generator=g)
    with torch.no_grad():
        teacher=torch.tanh(train@w1+b1)@w2+b2; eval_logits=torch.tanh(evalx@w1+b1)@w2+b2
        train_p=F.softmax(teacher,dim=-1);eval_p=F.softmax(eval_logits,dim=-1)
    return {'seed':seed,'w1':w1,'b1':b1,'w2':w2,'b2':b2,'train_x':train,'train_p':train_p,'eval_x':evalx,'eval_p':eval_p}


def fit(method:str,seed:int,w:dict[str,object])->tuple[SpecSystem,float,int]:
    sys=SpecSystem(method,w['w1'],w['b1'],w['w2'],w['b2']);params=list(sys.parameters());updates=UPDATES if params else 0
    start=time.perf_counter()
    if params:
        opt=torch.optim.Adam(params,lr=LR);g=torch.Generator().manual_seed(seed+7201)
        for _ in range(UPDATES):
            idx=torch.randint(w['train_x'].shape[0],(BATCH,),generator=g);x=w['train_x'][idx];p=w['train_p'][idx]
            q=F.log_softmax(sys.draft_logits(x),dim=-1);loss=F.kl_div(q,p,reduction='batchmean')
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    elapsed=time.perf_counter()-start;sys.eval();return sys,elapsed,updates


def score(sys:SpecSystem,w:dict[str,object])->dict[str,float]:
    x=w['eval_x'];p=w['eval_p']
    with torch.no_grad():
        q=F.softmax(sys.draft_logits(x),dim=-1)
        out,accept=speculative_distribution(q,p)
        kl=(p*(torch.log(p.clamp_min(1e-30))-torch.log(q.clamp_min(1e-30)))).sum(-1)
        return {'mean_acceptance_overlap':float(accept.mean()),'heldout_draft_to_full_KL':float(kl.mean()),
            'draft_full_top1_agreement':float((q.argmax(-1)==p.argmax(-1)).float().mean()),
            'corrected_distribution_max_abs_error':float((out-p).abs().max()),
            'corrected_distribution_mean_abs_error':float((out-p).abs().mean())}


def measure(fn,repeat:int=20)->float:
    for _ in range(3):fn()
    t=time.perf_counter()
    for _ in range(repeat):fn()
    return (time.perf_counter()-t)/repeat


def save_payload(path:Path,sys:SpecSystem)->tuple[int,str]:
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**sys.payload_arrays())
    raw=path.read_bytes();return len(raw),hashlib.sha256(raw).hexdigest()


def load_payload(path:Path)->SpecSystem:
    with np.load(path,allow_pickle=False) as z:return from_payload({k:z[k] for k in z.files})


def run(seed:int,condition:str,outdir:Path)->dict[str,object]:
    w=make_world(seed);rows=[];x=w['eval_x'];n=x.shape[0]
    full=SpecSystem('full_only',w['w1'],w['b1'],w['w2'],w['b2']).eval()
    verifier_s=measure(lambda:full.verifier_logits(x))/n
    for method in METHODS:
        t0=time.perf_counter();sys,train_s,updates=fit(method,seed,w)
        path=outdir/f'{condition}_{seed}_{method}.npz';size,digest=save_payload(path,sys)
        sys=load_payload(path);metrics=score(sys,w)
        draft_s=measure(lambda:sys.draft_logits(x))/n
        if method=='full_only':draft_s=verifier_s
        expected_tokens=1+metrics['mean_acceptance_overlap']
        spec_tps=expected_tokens/max(draft_s+verifier_s,1e-12)
        full_tps=1/max(verifier_s,1e-12)
        rows.append({'method':method,'payload_path':path.name,'serialized_bytes':size,'payload_sha256':digest,**metrics,
            **compute_proxy(method),'optimizer_updates':updates,'training_contexts_seen':updates*BATCH,
            'training_wall_s':train_s,'draft_latency_s_per_context':draft_s,'verifier_latency_s_per_context':verifier_s,
            'expected_tokens_per_cycle':expected_tokens,'expected_speculative_tokens_per_s':spec_tps,
            'full_only_tokens_per_s':full_tps,'speculative_speedup_over_full':spec_tps/full_tps,
            'total_wall_s':time.perf_counter()-t0})
    return {'experiment_id':'MA-399','condition':condition,'seed':seed,'train_contexts':2048,'eval_contexts':2048,
            'updates':UPDATES,'batch_size':BATCH,'learning_rate':LR,'results':rows}


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--condition',choices=('development','fresh'),required=True)
    p.add_argument('--outdir',type=Path,required=True);p.add_argument('--json',type=Path,required=True);a=p.parse_args()
    d=run(a.seed,a.condition,a.outdir);a.json.parent.mkdir(parents=True,exist_ok=True);a.json.write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps({'seed':a.seed,'condition':a.condition,'results':[{k:r[k] for k in ('method','serialized_bytes','mean_acceptance_overlap','corrected_distribution_max_abs_error','expected_speculative_tokens_per_s','speculative_speedup_over_full')} for r in d['results']]},indent=2))


if __name__=='__main__':main()
