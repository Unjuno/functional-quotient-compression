#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,platform,time
from pathlib import Path
import torch
from model import METHODS,VirtualLoRA,compute_proxy,task_id
ROOT=Path(__file__).resolve().parents[1];N,D,O,R,UPDATES,BATCH=8,16,12,2,1200,64
FIELDS=['condition','world_or_seed','mode','method','serialized_model_bytes','train_examples','optimizer_updates','active_compute_proxy','wall_time_s','inference_examples_per_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note']
def make_world(seed,mode):
 g=torch.Generator().manual_seed(seed)
 if mode=='shared_rotated_lora':return {'a':torch.randn(D,R,generator=g)*.5,'b':torch.randn(R,O,generator=g)*.5,'angle':torch.rand(N,generator=g)*6.283-3.1416,'aa':None,'bb':None}
 return {'a':None,'b':None,'angle':None,'aa':torch.randn(N,D,R,generator=g)*.5,'bb':torch.randn(N,R,O,generator=g)*.5}
def target(x,w,mode):
 if mode=='independent_lora':m=torch.einsum('ndr,nro->ndo',w['aa'],w['bb'])
 else:
  c,s=torch.cos(w['angle']),torch.sin(w['angle']);rot=torch.stack((torch.stack((c,-s),dim=-1),torch.stack((s,c),dim=-1)),dim=1);m=torch.einsum('dr,nrs,so->ndo',w['a'],rot,w['b'])
 return torch.einsum('bd,bdo->bo',x,m[task_id(x)])
def fit(method,w,mode,init,ds,lr):
 torch.manual_seed(init);m=VirtualLoRA(method,init);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(ds+37);t=time.perf_counter();m.train()
 for _ in range(UPDATES):
  x=torch.randn(BATCH,D,generator=g);y=target(x,w,mode);loss=(m(x)-y).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m,time.perf_counter()-t
def evaluate(m,w,mode,seed):
 x=torch.randn(8192,D,generator=torch.Generator().manual_seed(seed));y=target(x,w,mode);m.eval()
 with torch.no_grad():p=m(x);mse=float((p-y).square().mean());worst=max(float((p[task_id(x)==i]-y[task_id(x)==i]).square().mean()) for i in range(N));r2=float(1-(p-y).square().sum()/(y-y.mean()).square().sum())
 return mse,worst,r2,x
def throughput(m,x):
 x=x[:64]
 with torch.no_grad():
  for _ in range(4):m(x)
  t=time.perf_counter()
  for _ in range(20):m(x)
 return 64*20/(time.perf_counter()-t)
def main():
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['dev','fresh'],required=True);a.add_argument('--lr',type=float);z=a.parse_args();torch.set_num_threads(1);modes=['shared_rotated_lora','independent_lora']
 if z.phase=='dev':worlds,lrs=[(240000,2400000)],[.003,.01]
 else:
  if z.lr not in (.003,.01):raise SystemExit('fresh requires frozen LR')
  worlds,lrs=[(240001,2400001),(240002,2400002),(240003,2400003)],[z.lr]
 rows,scores=[],{lr:[] for lr in lrs}
 for wid,init in worlds:
  for mode in modes:
   w=make_world(wid,mode);ds=init+(0 if mode==modes[0] else 10000)
   for lr in lrs:
    for i,method in enumerate(METHODS):
     m,elapsed=fit(method,w,mode,init+i*193+int(lr*10000),ds,lr);mse,worst,r2,x=evaluate(m,w,mode,init+91)
     if z.phase=='dev':scores[lr].append(mse)
     row={'condition':z.phase,'world_or_seed':wid,'mode':mode,'method':method,'serialized_model_bytes':m.serialized_payload_bytes(),'train_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':compute_proxy(method,UPDATES*BATCH),'wall_time_s':round(elapsed,6),'inference_examples_per_s':round(throughput(m,x),3),'primary_metric':'adapter_MSE','primary_value':f'{mse:.10g}','secondary_metric':'worst_task_MSE;R2','secondary_value':f'{worst:.10g};{r2:.8g}','status_note':f'lr={lr}; matched minibatches; oracle task id'};rows.append(row);print(z.phase,wid,mode,method,lr,'MSE',mse,'worst',worst,'bytes',row['serialized_model_bytes'],flush=True)
 if z.phase=='dev':
  best=min(scores,key=lambda lr:sum(scores[lr])/len(scores[lr]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-024','selected_common_lr':best,'mean_mse_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'fresh_worlds':[240001,240002,240003],'updates':UPDATES},indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writerows(rows)
 print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),'rows':len(rows)}))
if __name__=='__main__':main()
