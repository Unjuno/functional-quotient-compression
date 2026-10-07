#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,platform,time
from pathlib import Path
import torch
from model import METHODS,ExpandedAttention,compute_proxy,givens,H,P,D,HD,L,GROUP
ROOT=Path(__file__).resolve().parents[1];UPDATES,BATCH=600,2
FIELDS=['condition','world_or_seed','mode','method','serialized_model_bytes','train_examples','optimizer_updates','active_compute_proxy','wall_time_s','inference_examples_per_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note']
def make_world(seed,mode):
 g=torch.Generator().manual_seed(seed);out=torch.randn(D,D,generator=g)*.06
 if mode=='shared4_view16':return {'w':torch.randn(P,3,D,HD,generator=g)*.22,'angle':torch.rand(H,D//2,generator=g)*.7-.35,'full':None,'out':out}
 return {'w':None,'angle':None,'full':torch.randn(H,3,D,HD,generator=g)*.22,'out':out}
def head_contrib(x,w,mode):
 if mode=='shared4_view16':
  xv=torch.stack([givens(x,a.expand(x.shape[0],x.shape[1],-1)) for a in w['angle']],dim=1);qkv=torch.einsum('bhld,hkdm->bhklm',xv,w['w'][GROUP])
 else:qkv=torch.einsum('bld,hkdm->bhklm',x,w['full'])
 q,k,v=qkv[:,:,0],qkv[:,:,1],qkv[:,:,2];att=torch.softmax(torch.einsum('bhlm,bhtm->bhlt',q,k)/(HD**.5),dim=-1);ctx=torch.einsum('bhlt,bhtm->bhlm',att,v).transpose(1,2)
 return torch.stack([ctx[:,:,h]@w['out'][h*HD:(h+1)*HD] for h in range(H)],dim=2)
def target(x,w,mode):return head_contrib(x,w,mode).sum(dim=2)
def fit(method,w,mode,init,ds,lr):
 torch.manual_seed(init);m=ExpandedAttention(method,init);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(ds+47);t=time.perf_counter();m.train()
 for _ in range(UPDATES):
  x=torch.randn(BATCH,L,D,generator=g);y=target(x,w,mode);loss=(m(x)-y).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m,time.perf_counter()-t
def evaluate(m,w,mode,seed):
 x=torch.randn(128,L,D,generator=torch.Generator().manual_seed(seed));y=target(x,w,mode);m.eval()
 with torch.no_grad():p=m(x);tc=head_contrib(x,w,mode);mc=m.head_contributions(x);mse=float((p-y).square().mean());worst=max(float((mc[:,:,h]-tc[:,:,h]).square().mean()) for h in range(H));r2=float(1-(p-y).square().sum()/(y-y.mean()).square().sum())
 return mse,worst,r2,x
def throughput(m,x):
 with torch.no_grad():
  for _ in range(2):m(x)
  t=time.perf_counter()
  for _ in range(8):m(x)
 return x.shape[0]*8/(time.perf_counter()-t)
def main():
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['dev','fresh'],required=True);a.add_argument('--lr',type=float);z=a.parse_args();torch.set_num_threads(1);modes=['shared4_view16','independent16']
 if z.phase=='dev':worlds,lrs=[(48000,480000)],[.001,.003]
 else:
  if z.lr not in (.001,.003):raise SystemExit('fresh requires frozen LR')
  worlds,lrs=[(48001,480001),(48002,480002),(48003,480003)],[z.lr]
 rows,scores=[],{lr:[] for lr in lrs}
 for wid,init in worlds:
  for mode in modes:
   w=make_world(wid,mode);ds=init+(0 if mode==modes[0] else 10000)
   for lr in lrs:
    for i,method in enumerate(METHODS):
     m,elapsed=fit(method,w,mode,init+i*193+int(lr*100000),ds,lr);mse,worst,r2,x=evaluate(m,w,mode,init+91)
     if z.phase=='dev':scores[lr].append(mse)
     row={'condition':z.phase,'world_or_seed':wid,'mode':mode,'method':method,'serialized_model_bytes':m.serialized_payload_bytes(),'train_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':compute_proxy(method,UPDATES*BATCH),'wall_time_s':round(elapsed,6),'inference_examples_per_s':round(throughput(m,x),3),'primary_metric':'attention_output_MSE','primary_value':f'{mse:.10g}','secondary_metric':'worst_logical_head_contribution_MSE;R2','secondary_value':f'{worst:.10g};{r2:.8g}','status_note':f'lr={lr}; matched minibatches; physical=4 logical=16'};rows.append(row);print(z.phase,wid,mode,method,lr,'MSE',mse,'worst_head',worst,'bytes',row['serialized_model_bytes'],flush=True)
 if z.phase=='dev':
  best=min(scores,key=lambda lr:sum(scores[lr])/len(scores[lr]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-048','selected_common_lr':best,'mean_mse_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'fresh_worlds':[48001,48002,48003],'updates':UPDATES},indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writerows(rows)
 print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),'rows':len(rows)}))
if __name__=='__main__':main()
