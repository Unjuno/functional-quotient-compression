#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,platform,time
from pathlib import Path
import torch
from model import METHODS,AttentionHeads,compute_proxy,givens,H,D,HD,L
ROOT=Path(__file__).resolve().parents[1];UPDATES,BATCH=900,8
FIELDS=['condition','world_or_seed','mode','method','serialized_model_bytes','train_examples','optimizer_updates','active_compute_proxy','wall_time_s','inference_examples_per_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note']
def make_world(seed,mode):
 g=torch.Generator().manual_seed(seed);out=torch.randn(D,D,generator=g)*.08
 if mode=='shared_qkv_views':return {'w':torch.randn(3,D,HD,generator=g)*.25,'angle':torch.rand(H,D//2,generator=g)*.8-.4,'full':None,'out':out}
 return {'w':None,'angle':None,'full':torch.randn(H,3,D,HD,generator=g)*.25,'out':out}
def head_contrib(x,w,mode):
 if mode=='shared_qkv_views':
  xv=torch.stack([givens(x,a.expand(x.shape[0],x.shape[1],-1)) for a in w['angle']],dim=1);qkv=torch.einsum('bhld,kdm->bhklm',xv,w['w'])
 else:qkv=torch.einsum('bld,hkdm->bhklm',x,w['full'])
 q,k,v=qkv[:,:,0],qkv[:,:,1],qkv[:,:,2];a=torch.softmax(torch.einsum('bhlm,bhtm->bhlt',q,k)/(HD**.5),dim=-1);c=torch.einsum('bhlt,bhtm->bhlm',a,v)
 ct=c.transpose(1,2);return torch.stack([ct[:,:,i]@w['out'][i*HD:(i+1)*HD] for i in range(H)],dim=2)
def target(x,w,mode):return head_contrib(x,w,mode).sum(dim=2)
def fit(method,w,mode,init,ds,lr):
 torch.manual_seed(init);m=AttentionHeads(method,init);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(ds+41);t=time.perf_counter();m.train()
 for _ in range(UPDATES):
  x=torch.randn(BATCH,L,D,generator=g);y=target(x,w,mode);loss=(m(x)-y).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m,time.perf_counter()-t
def evaluate(m,w,mode,seed):
 x=torch.randn(256,L,D,generator=torch.Generator().manual_seed(seed));y=target(x,w,mode);m.eval()
 with torch.no_grad():p=m(x);mse=float((p-y).square().mean());tc=head_contrib(x,w,mode);mc=m.head_contributions(x);worst=max(float((mc[:,:,i]-tc[:,:,i]).square().mean()) for i in range(H));r2=float(1-(p-y).square().sum()/(y-y.mean()).square().sum())
 return mse,worst,r2,x
def throughput(m,x):
 with torch.no_grad():
  for _ in range(3):m(x)
  t=time.perf_counter()
  for _ in range(10):m(x)
 return x.shape[0]*10/(time.perf_counter()-t)
def main():
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['dev','fresh'],required=True);a.add_argument('--lr',type=float);z=a.parse_args();torch.set_num_threads(1);modes=['shared_qkv_views','independent_qkv']
 if z.phase=='dev':worlds,lrs=[(41000,410000)],[.001,.003]
 else:
  if z.lr not in (.001,.003):raise SystemExit('fresh requires frozen LR')
  worlds,lrs=[(41001,410001),(41002,410002),(41003,410003)],[z.lr]
 rows,scores=[],{lr:[] for lr in lrs}
 for wid,init in worlds:
  for mode in modes:
   w=make_world(wid,mode);ds=init+(0 if mode==modes[0] else 10000)
   for lr in lrs:
    for i,method in enumerate(METHODS):
     m,elapsed=fit(method,w,mode,init+i*193+int(lr*100000),ds,lr);mse,worst,r2,x=evaluate(m,w,mode,init+91)
     if z.phase=='dev':scores[lr].append(mse)
     row={'condition':z.phase,'world_or_seed':wid,'mode':mode,'method':method,'serialized_model_bytes':m.serialized_payload_bytes(),'train_examples':UPDATES*BATCH*L,'optimizer_updates':UPDATES,'active_compute_proxy':compute_proxy(method,UPDATES*BATCH*L),'wall_time_s':round(elapsed,6),'inference_examples_per_s':round(throughput(m,x),3),'primary_metric':'attention_output_MSE','primary_value':f'{mse:.10g}','secondary_metric':'worst_head_contribution_MSE;R2','secondary_value':f'{worst:.10g};{r2:.8g}','status_note':f'lr={lr}; matched minibatches; full attention; shared output projection'};rows.append(row);print(z.phase,wid,mode,method,lr,'MSE',mse,'worst_head',worst,'bytes',row['serialized_model_bytes'],flush=True)
 if z.phase=='dev':
  best=min(scores,key=lambda lr:sum(scores[lr])/len(scores[lr]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-041','selected_common_lr':best,'mean_mse_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'fresh_worlds':[41001,41002,41003],'updates':UPDATES},indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writerows(rows)
 print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),'rows':len(rows)}))
if __name__=='__main__':main()
