#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,platform,time
from pathlib import Path
import torch
from model import METHODS,CausalKV,cache_bytes,compute_proxy,rotate_heads,H,D,HD,L
ROOT=Path(__file__).resolve().parents[1];UPDATES,BATCH=600,8
FIELDS=['condition','world_or_seed','mode','method','serialized_model_bytes','cache_bytes_per_sequence','logical_expanded_cache_bytes','train_examples','optimizer_updates','active_compute_proxy','wall_time_s','inference_examples_per_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note']
def make_world(seed,mode):
 g=torch.Generator().manual_seed(seed);q=torch.randn(H,D,HD,generator=g)*.24;o=torch.randn(D,D,generator=g)*.08
 if mode=='shared_causal_kv':return {'q':q,'k':torch.randn(D,HD,generator=g)*.24,'v':torch.randn(D,HD,generator=g)*.24,'ka':torch.rand(H,HD//2,generator=g)*.7-.35,'va':torch.rand(H,HD//2,generator=g)*.7-.35,'kf':None,'vf':None,'out':o}
 return {'q':q,'k':None,'v':None,'ka':None,'va':None,'kf':torch.randn(H,D,HD,generator=g)*.24,'vf':torch.randn(H,D,HD,generator=g)*.24,'out':o}
def head_contributions(x,w,mode):
 q=torch.einsum('bld,hdm->bhlm',x,w['q'])
 if mode=='shared_causal_kv':
  k=torch.einsum('bld,dm->blm',x,w['k'])[:,None].expand(-1,H,-1,-1);v=torch.einsum('bld,dm->blm',x,w['v'])[:,None].expand(-1,H,-1,-1);k=rotate_heads(k,w['ka']);v=rotate_heads(v,w['va'])
 else:k=torch.einsum('bld,hdm->bhlm',x,w['kf']);v=torch.einsum('bld,hdm->bhlm',x,w['vf'])
 from model import causal_weights
 a=causal_weights(q,k);c=torch.einsum('bhlt,bhtm->bhlm',a,v).transpose(1,2)
 return torch.stack([c[:,:,h]@w['out'][h*HD:(h+1)*HD] for h in range(H)],dim=2)
def target(x,w,mode):return head_contributions(x,w,mode).sum(dim=2)
def fit(method,w,mode,init,ds,lr):
 torch.manual_seed(init);m=CausalKV(method,init);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(ds+59);t=time.perf_counter();m.train()
 for _ in range(UPDATES):
  x=torch.randn(BATCH,L,D,generator=g);y=target(x,w,mode);loss=(m(x)-y).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m,time.perf_counter()-t
def evaluate(m,w,mode,seed):
 x=torch.randn(128,L,D,generator=torch.Generator().manual_seed(seed));y=target(x,w,mode);m.eval()
 with torch.no_grad():p=m(x);tc=head_contributions(x,w,mode);mc=m.head_contributions(x);mse=float((p-y).square().mean());worst=max(float((mc[:,:,h]-tc[:,:,h]).square().mean()) for h in range(H));r2=float(1-(p-y).square().sum()/(y-y.mean()).square().sum())
 return mse,worst,r2,x
def throughput(m,x):
 with torch.no_grad():
  for _ in range(3):m(x)
  t=time.perf_counter()
  for _ in range(10):m(x)
 return x.shape[0]*10/(time.perf_counter()-t)
def main():
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['dev','fresh'],required=True);a.add_argument('--lr',type=float);z=a.parse_args();torch.set_num_threads(1);modes=['shared_causal_kv','independent_causal_kv']
 if z.phase=='dev':worlds,lrs=[(63000,630000)],[.001,.003]
 else:
  if z.lr not in (.001,.003):raise SystemExit('fresh requires frozen LR')
  worlds,lrs=[(63001,630001),(63002,630002),(63003,630003)],[z.lr]
 rows,scores=[],{lr:[] for lr in lrs}
 for wid,init in worlds:
  for mode in modes:
   w=make_world(wid,mode);ds=init+(0 if mode==modes[0] else 10000)
   for lr in lrs:
    for i,method in enumerate(METHODS):
     m,elapsed=fit(method,w,mode,init+i*193+int(lr*100000),ds,lr);mse,worst,r2,x=evaluate(m,w,mode,init+91)
     if z.phase=='dev':scores[lr].append(mse)
     cache=cache_bytes(method,1,L);logical=2*H*HD*L*4
     row={'condition':z.phase,'world_or_seed':wid,'mode':mode,'method':method,'serialized_model_bytes':m.serialized_payload_bytes(),'cache_bytes_per_sequence':cache,'logical_expanded_cache_bytes':logical,'train_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':compute_proxy(method,UPDATES*BATCH),'wall_time_s':round(elapsed,6),'inference_examples_per_s':round(throughput(m,x),3),'primary_metric':'causal_attention_output_MSE','primary_value':f'{mse:.10g}','secondary_metric':'worst_head_contribution_MSE;R2','secondary_value':f'{worst:.10g};{r2:.8g}','status_note':f'lr={lr}; matched minibatches; causal; cache f32; seq={L}'};rows.append(row);print(z.phase,wid,mode,method,lr,'MSE',mse,'cacheB',cache,'payloadB',row['serialized_model_bytes'],flush=True)
 if z.phase=='dev':
  best=min(scores,key=lambda lr:sum(scores[lr])/len(scores[lr]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-063','selected_common_lr':best,'mean_mse_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'fresh_worlds':[63001,63002,63003],'updates':UPDATES},indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writerows(rows)
 print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),'rows':len(rows)}))
if __name__=='__main__':main()
