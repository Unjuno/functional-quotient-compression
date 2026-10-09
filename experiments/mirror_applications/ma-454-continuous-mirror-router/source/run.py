#!/usr/bin/env python3
"""Small deterministic MA-454 continuous context router screen."""
import csv, hashlib, io, json, math, statistics, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'; PAY=ART/'payloads'
DEV=[45400,45401]; FRESH=[45420,45421,45422]; SEEDS=[0,1,2]; K=8

def task(w,s,tid):
 g=torch.Generator().manual_seed(w*1000003+s*997+tid*53)
 a=.5+1.5*torch.rand((),generator=g); b=-.5+torch.rand((),generator=g)
 x=-2+4*torch.rand(32,generator=g); y=torch.tanh(a*x+b)
 qx=-2+4*torch.rand(256,generator=torch.Generator().manual_seed(w*99131+s*71+tid*17))
 return float(a),float(b),x,y,qx,torch.tanh(a*qx+b)
def ctx(x,y): return torch.stack([x.mean(),y.mean(),(x*y).mean(),y.square().mean()])
def proto(points,k):
 c=points[torch.linspace(0,len(points)-1,k).long()].clone()
 for _ in range(50):
  ids=torch.cdist(points,c).argmin(1)
  for j in range(k):
   if (ids==j).any(): c[j]=points[ids==j].mean(0)
 return c
def fit_hyper(contexts,targets,lam):
 # Ridge map, with intercept. This is algebraically the same family as the continuous Mirror router.
 X=torch.cat([contexts,torch.ones(len(contexts),1)],1)
 reg=torch.eye(X.shape[1])*lam; reg[-1,-1]=0
 return torch.linalg.solve(X.T@X+reg,X.T@targets)
def pred_code(ctxv,W,mean,std): return torch.cat([(ctxv-mean)/std,torch.ones(1)])@W
def nrmse(p,y): return float(((p-y).square().mean().sqrt())/(y.square().mean().sqrt()+1e-12))
def package(method,N,weights,centers,indices,codes,contexts):
 # Same wire representation for Mirror and generic hypernetwork: the mapping is algebraically identical.
 obj={'format':'ma454-v1','method':('conditional_linear' if method in ('mirror','hyper') else method),'N':N,'shared':'tanh'}
 if method in ('mirror','hyper'): obj.update(router=weights,context_norm=torch.tensor([0.,1.]))
 if method=='discrete': obj.update(prototypes=centers,route_indices=torch.tensor(indices[:N],dtype=torch.uint8))
 if method=='mirror': obj['context_codes']=torch.stack(contexts[:N])
 if method=='hyper': obj['context_codes']=torch.stack(contexts[:N])
 if method=='independent': obj['private_codes']=torch.stack(codes[:N])
 if method=='shared': pass
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','fresh'],required=True);a=ap.parse_args();ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True)
 if a.phase=='development':
  cs=[];ts=[]; pts=[]
  for w in DEV:
   for s in SEEDS:
    for tid in range(100,164):
     aa,bb,x,y,qx,qy=task(w,s,tid);cs.append(ctx(x,y));ts.append(torch.tensor([aa,bb]));pts.append([aa,bb])
  C=torch.stack(cs); T=torch.stack(ts); P=torch.tensor(pts); best=None; trials=[]
  for norm in ['raw','standardized']:
   mean=C.mean(0) if norm=='standardized' else torch.zeros(C.shape[1]); std=C.std(0).clamp_min(1e-6) if norm=='standardized' else torch.ones(C.shape[1]); Cn=(C-mean)/std
   for lam in [.001,.01,.1,1.0]:
    W=fit_hyper(Cn,T,lam); pred=torch.cat([Cn,torch.ones(len(Cn),1)],1)@W
    score=float(((pred-T).square().mean()).sqrt()); trials.append({'norm':norm,'lambda':lam,'rmse':score})
    if best is None or score<best[0]: best=(score,lam,norm,W,mean,std)
  centers=proto(P,K)
  torch.save({'W':best[3],'centers':centers,'mean':best[4],'std':best[5]},PAY/'frozen_router.pt')
  (ART/'development_selection.json').write_text(json.dumps({'ridge_lambda':best[1],'normalization':best[2],'dev_code_rmse':best[0],'trials':trials,'K':K,'fresh_untouched':True},indent=2)+'\n')
  print(json.dumps({'ridge_lambda':best[1],'normalization':best[2],'dev_code_rmse':best[0]}));return
 frozen=torch.load(PAY/'frozen_router.pt',weights_only=True); W=frozen['W']; centers=frozen['centers']; mean=frozen['mean']; std=frozen['std']; rows=[]
 for w in FRESH:
  for s in SEEDS:
   records=[]
   for tid in range(5000,5064):
    aa,bb,x,y,qx,qy=task(w,s,tid); c=ctx(x,y); pred=pred_code(c,W,mean,std); losses=[float(torch.nn.functional.mse_loss(torch.tanh(c0*x[0:1]+c1),y[0:1])) for c0,c1 in centers]
    # Discrete router uses support loss against all 32 observations.
    losses=[float(torch.nn.functional.mse_loss(torch.tanh(c0*x+c1),y)) for c0,c1 in centers]; ix=min(range(K),key=lambda i:losses[i]); t=time.perf_counter(); mirror_out=torch.tanh(pred[0]*qx+pred[1]); wm=time.perf_counter()-t
    t=time.perf_counter(); hyper_out=torch.tanh(pred[0]*qx+pred[1]); wh=time.perf_counter()-t
    t=time.perf_counter(); discrete_out=torch.tanh(centers[ix,0]*qx+centers[ix,1]); wd=time.perf_counter()-t
    t=time.perf_counter(); ind=torch.tensor([aa,bb]); indep_out=torch.tanh(ind[0]*qx+ind[1]); wi=time.perf_counter()-t
    shared_out=torch.tanh(qx)
    records.append({'c':c,'code':pred,'index':ix,'true':ind,'scores':{'shared':nrmse(shared_out,qy),'mirror':nrmse(mirror_out,qy),'hyper':nrmse(hyper_out,qy),'discrete':nrmse(discrete_out,qy),'independent':nrmse(indep_out,qy)},'wall':{'mirror':wm,'hyper':wh,'discrete':wd,'independent':wi}})
   for method in ['shared','discrete','mirror','hyper','independent']:
    for N in [1,20,64]:
     contexts=[r['c'] for r in records]; codes=[r['true'] for r in records]; data=package(method,N,W,centers,[r['index'] for r in records],codes,contexts); path=PAY/f'{w}_{s}_{method}_N{N}.pt';path.write_bytes(data)
     rows.append({'world':w,'seed':s,'method':method,'n':N,'tasks':N,'nrmse_mean':statistics.mean(r['scores'][method] for r in records[:N]),'payload_bytes':len(data),'bytes_per_task':len(data)/N,'compute_MAC_per_task':({'shared':0,'discrete':K*32,'mirror':4*2+2*256,'hyper':4*2+2*256,'independent':0}[method]),'query_wall_seconds_mean':(0.0 if method=='shared' else statistics.mean(r['wall'][method] for r in records[:N])),'hash':hashlib.sha256(data).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with open(ART/'fresh_runs.jsonl','w') as f:f.write(''.join(json.dumps(r)+'\n' for r in rows))
 with open(ROOT/'RESULTS_CORE.csv','w',newline='') as f:wri=csv.DictWriter(f,fieldnames=list(rows[0]));wri.writeheader();wri.writerows(rows)
 print(json.dumps({'phase':'fresh','rows':len(rows)}))
if __name__=='__main__':main()
