#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,platform,time
from pathlib import Path
import torch
from model import METHODS,ExpertBasis,compute_proxy,role_of
ROOT=Path(__file__).resolve().parents[1];D,O,N,UPDATES,BATCH=16,12,8,1200,64
FIELDS=['condition','world_or_seed','mode','method','serialized_model_bytes','train_examples','optimizer_updates','active_compute_proxy','wall_time_s','inference_examples_per_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note']
def make_world(seed,mode):
 g=torch.Generator().manual_seed(seed)
 if mode=='unit_circle_basis':return {'basis':torch.randn(2,D,O,generator=g)*.6,'coef':torch.rand(N,generator=g)*6.283-3.1416,'w':None}
 return {'basis':None,'coef':None,'w':torch.randn(N,D,O,generator=g)*.6}
def target(x,w,mode):
 if mode=='unit_circle_basis':
  c,s=torch.cos(w['coef']),torch.sin(w['coef']);m=c[:,None,None]*w['basis'][0]+s[:,None,None]*w['basis'][1]
 else:m=w['w']
 return torch.einsum('bd,bdo->bo',x,m[role_of(x)])
def fit(method,w,mode,init,ds,lr):
 torch.manual_seed(init);m=ExpertBasis(method,init);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(ds+29);t=time.perf_counter();m.train()
 for _ in range(UPDATES):
  x=torch.randn(BATCH,D,generator=g);y=target(x,w,mode);loss=(m(x)-y).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m,time.perf_counter()-t
def evaluate(m,w,mode,seed):
 g=torch.Generator().manual_seed(seed);x=torch.randn(8192,D,generator=g);y=target(x,w,mode);m.eval()
 with torch.no_grad():p=m(x);mse=float((p-y).square().mean());r2=float(1-(p-y).square().sum()/(y-y.mean()).square().sum());errs=[]
 for i in range(N):mask=role_of(x)==i;errs.append(float((p[mask]-y[mask]).square().mean()))
 r2=float(1-(p-y).square().sum()/(y-y.mean()).square().sum())
 return mse,max(errs),r2,x
def throughput(m,x):
 x=x[:64]
 with torch.no_grad():
  for _ in range(4):m(x)
  t=time.perf_counter()
  for _ in range(20):m(x)
 return 64*20/(time.perf_counter()-t)
def main():
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['dev','fresh'],required=True);a.add_argument('--lr',type=float);z=a.parse_args();torch.set_num_threads(1);modes=['unit_circle_basis','independent_experts']
 if z.phase=='dev':worlds,lrs=[(190000,1900000)],[.003,.01]
 else:
  if z.lr not in (.003,.01):raise SystemExit('fresh requires frozen LR')
  worlds,lrs=[(190001,1900001),(190002,1900002),(190003,1900003)],[z.lr]
 rows,scores=[],{lr:[] for lr in lrs}
 for wid,init in worlds:
  for mode in modes:
   w=make_world(wid,mode);ds=init+(0 if mode==modes[0] else 10000)
   for lr in lrs:
    for i,method in enumerate(METHODS):
     m,elapsed=fit(method,w,mode,init+i*193+int(lr*10000),ds,lr);mse,worst,r2,x=evaluate(m,w,mode,init+91)
     if z.phase=='dev':scores[lr].append(mse)
     row={'condition':z.phase,'world_or_seed':wid,'mode':mode,'method':method,'serialized_model_bytes':m.serialized_payload_bytes(),'train_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':compute_proxy(method,UPDATES*BATCH),'wall_time_s':round(elapsed,6),'inference_examples_per_s':round(throughput(m,x),3),'primary_metric':'routed_MSE','primary_value':f'{mse:.10g}','secondary_metric':'worst_role_MSE;R2','secondary_value':f'{worst:.10g};{r2:.8g}','status_note':f'lr={lr}; matched minibatches; oracle hard route'}
     rows.append(row);print(z.phase,wid,mode,method,lr,'MSE',mse,'worst',worst,'bytes',row['serialized_model_bytes'],flush=True)
 if z.phase=='dev':
  best=min(scores,key=lambda lr:sum(scores[lr])/len(scores[lr]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-019','selected_common_lr':best,'mean_mse_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'fresh_worlds':[190001,190002,190003],'updates':UPDATES},indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writerows(rows)
 print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),'rows':len(rows)}))
if __name__=='__main__':main()
