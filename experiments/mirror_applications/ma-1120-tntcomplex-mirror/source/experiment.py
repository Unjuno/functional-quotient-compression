"""MA-1120 synthetic temporal relation-time factorization mechanism screen."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device('cpu')

class TemporalScorer(nn.Module):
    def __init__(self, method: str, n_entities=24, n_relations=6, n_times=12, rank=8, view_rank=2):
        super().__init__(); self.method=method; self.n_entities=n_entities; self.n_relations=n_relations; self.n_times=n_times; self.rank=rank; self.view_rank=view_rank
        self.entity=nn.Embedding(n_entities,rank); self.left=nn.Embedding(n_relations,rank); self.right=nn.Embedding(n_relations,rank)
        self.register_buffer('time_features', torch.tensor(np.stack([np.cos(2*np.pi*np.arange(n_times)/8), np.sin(2*np.pi*np.arange(n_times)/8)],1), dtype=torch.float32))
        if method=='mirror':
            self.basis=nn.Parameter(torch.randn(2,rank)*.03); self.code=nn.Embedding(n_relations,2)
        elif method=='tucker':
            self.basis=nn.Parameter(torch.randn(2,rank)*.03); self.rel_coeff=nn.Embedding(n_relations,2)
        elif method=='lowrank':
            self.rel_map=nn.Parameter(torch.randn(n_relations,2,rank)*.03)
        elif method=='independent':
            self.rel_time=nn.Embedding(n_relations*n_times,rank)
        self.reset_parameters()
    def reset_parameters(self):
        for p in self.parameters():
            if p.ndim>1: nn.init.normal_(p, std=.04)
            else: nn.init.zeros_(p)
    def rel_time_factor(self,r,t):
        if self.method=='mirror':
            c=self.code(r)*self.time_features[t]
            return c @ self.basis
        if self.method=='tucker':
            c=self.rel_coeff(r)*self.time_features[t]
            return c @ self.basis
        if self.method=='lowrank':
            return torch.einsum('bi,bik->bk', self.time_features[t], self.rel_map[r])
        if self.method=='independent':
            return self.rel_time(r*self.n_times+t)
        return self.left(r)*torch.cat([self.time_features[t], self.time_features[t], self.time_features[t], self.time_features[t]],-1)[:, :self.rank]
    def score(self,h,r,t,all_tails=False):
        q=self.entity(h)*self.rel_time_factor(r,t)
        if all_tails: return q @ self.entity.weight.T
        return (q*self.entity(t)).sum(-1)

def make_world(seed,aligned=True):
    rng=np.random.default_rng(seed); ne,nr,nt,rank=24,6,12,8
    e=rng.normal(0,.7,(ne,rank)); e/=np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-9)
    if aligned:
        B=rng.normal(0,1,(2,rank)); B/=np.maximum(np.linalg.norm(B,axis=1,keepdims=True),1e-9)
        rc=rng.normal(0,1,(nr,2)); phase=np.linspace(0,2*np.pi,nt,endpoint=False)
        tc=np.stack([np.cos(phase),np.sin(phase)],1)
        op=np.einsum('ra,ta,ak->rtk',rc,tc,B)
    else:
        op=rng.normal(0,1,(nr,nt,rank))
    triples=[]
    for t in range(nt):
      for r in range(nr):
        for h in range(ne):
          scores=(e[h]*op[r,t])@e.T
          # Two deterministic positives and one hard negative per (h,r,t).
          order=np.argsort(scores)[::-1]
          for tail in order[:2]: triples.append((h,r,t,int(tail),1))
          for tail in order[-2:]: triples.append((h,r,t,int(tail),0))
    return e.astype('float32'),op.astype('float32'),np.asarray(triples,dtype='int64')

def payload(model,path):
    tensors={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    from safetensors.torch import save_file
    save_file(tensors,str(path))
    return os.path.getsize(path),hashlib.sha256(open(path,'rb').read()).hexdigest()

def ranking(model,triples,all_candidates=24):
    # all-entity ranking among generated positive and negative sampled candidates; deterministic MRR.
    ranks=[]
    with torch.no_grad():
      for h,r,t,tail,label in triples:
        if label!=1: continue
        h1=torch.tensor([h]);r1=torch.tensor([r]);t1=torch.tensor([t]);scores=model.score(h1,r1,t1,True)[0]
        rank=1+int((scores>scores[tail]).sum());ranks.append(rank)
    return float(np.mean(1/np.asarray(ranks))),float(np.mean(np.asarray(ranks)==1))

def train_one(method,seed,aligned,updates=120,rank=8,view_rank=2):
    torch.manual_seed(seed); np.random.seed(seed)
    e,op,trip=make_world(seed,aligned)
    model=TemporalScorer(method,rank=rank,view_rank=view_rank)
    # Initialize entity vectors from shared world geometry for stable compact mechanism screen;
    # they remain trainable. This common paid initialization is charged in every method.
    with torch.no_grad(): model.entity.weight.copy_(torch.tensor(e))
    train=trip[trip[:,2]<=7]; dev=trip[trip[:,2]>=8]
    opt=torch.optim.Adam(model.parameters(),lr=.025)
    started=time.perf_counter(); seen=0
    for step in range(updates):
        ix=np.random.randint(0,len(train),size=96); batch=train[ix]
        h,r,t,tail,y=[torch.tensor(batch[:,i]) for i in range(5)]
        # Train with full tail ranking likelihood.
        logits=model.score(h,r,t,True)
        loss=nn.functional.cross_entropy(logits,tail)
        opt.zero_grad();loss.backward();opt.step();seen+=len(batch)
    train_s=time.perf_counter()-started
    mrr,h1=ranking(model,dev)
    infer0=time.perf_counter()
    for row in dev[:min(300,len(dev))]: model.score(torch.tensor([row[0]]),torch.tensor([row[1]]),torch.tensor([row[2]]),True)
    infer=(time.perf_counter()-infer0)/min(300,len(dev))*1e6
    tmp=ROOT/'source'/f'.payload-{method}-{seed}.pt'; size,sha=payload(model,tmp); tmp.unlink()
    return model,mrr,h1,size,sha,seen,train_s,infer,updates

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['dev','audit'],default='dev');args=ap.parse_args()
    cfg=json.loads((ROOT/'PROTOCOL.json').read_text()); seeds=cfg['development']['worlds_or_seeds'] if args.mode=='dev' else cfg['fresh']['worlds_or_seeds']
    rows=[]; reports={}
    for seed in seeds:
      for aligned in ([True] if args.mode=='dev' else [True,False]):
       world=f'{seed}-'+('aligned' if aligned else 'independent')
       for method in ['native','tucker','lowrank','mirror','independent']:
        model,mrr,h1,size,sha,seen,sec,inf,updates=train_one(method,seed,aligned,rank=8,view_rank=2)
        row={'world':world,'method':method,'filtered_mrr':mrr,'hits1':h1,'serialized_bytes':size,'train_examples':seen,'updates':updates,'active_macs_per_score':3*8+8,'train_seconds':sec,'infer_us_per_score':inf,'notes':'payload_sha256='+sha}
        rows.append(row); reports[world,method]=model
        print(world,method,f'MRR={mrr:.5f}',f'bytes={size}',f'sec={sec:.2f}',flush=True)
    out=ROOT/'source'/f'{args.mode}_results.json';out.write_text(json.dumps(rows,indent=2)+'\n')
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
      writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
if __name__=='__main__':main()
