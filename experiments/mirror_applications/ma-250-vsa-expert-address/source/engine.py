#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,platform,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import METHODS,ExpertViews,compute_proxy,givens
ROOT=Path(__file__).resolve().parents[1];R=4;D=16;O=12;UPDATES=1200;FIELDS=['condition','world_or_seed','mode','method','serialized_model_bytes','train_examples','optimizer_updates','active_compute_proxy','wall_time_s','inference_examples_per_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note']
def make_world(seed,mode):
 g=torch.Generator().manual_seed(seed);w=torch.randn(D,O,generator=g)*.6;b=torch.randn(O,generator=g)*.1;a=torch.rand(R,D//2,generator=g)*1.4-.7
 if mode=='independent_roles':wi=torch.randn(R,D,O,generator=g)*.6;bi=torch.randn(R,O,generator=g)*.1
 else:wi=bi=None
 return {'w':w,'b':b,'angles':a,'wi':wi,'bi':bi}
def target(x,role,w,mode):
 if mode=='aligned_shared_base':h=givens(x,w['angles'][role]);return h@w['w']+w['b']
 return torch.einsum('bd,bod->bo',x,w['wi'][role].transpose(1,2))+w['bi'][role]
def evaluate(m,w,mode,seed):
 g=torch.Generator().manual_seed(seed);x=torch.randn(4096,D,generator=g);role=torch.randint(R,(4096,),generator=g);y=target(x,role,w,mode);m.eval()
 with torch.no_grad():pred=m(x,role);mse=float((pred-y).square().mean());r2=float(1-(pred-y).square().sum()/(y-y.mean()).square().sum())
 return mse,r2,x,role
def fit(method,w,mode,seed,data_seed,lr):
 torch.manual_seed(seed);m=ExpertViews(method,data_seed,R,D,O);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(data_seed+31);start=time.perf_counter();m.train()
 for _ in range(UPDATES):
  x=torch.randn(64,D,generator=g);role=torch.randint(R,(64,),generator=g);y=target(x,role,w,mode);pred=m(x,role);loss=F.mse_loss(pred,y);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m,time.perf_counter()-start
def throughput(m,x,role):
 x=x[:64];role=role[:64];m.eval()
 with torch.no_grad():
  for _ in range(5):m(x,role)
  t=time.perf_counter()
  for _ in range(30):m(x,role)
 return 64*30/(time.perf_counter()-t)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);ap.add_argument('--lr',type=float);a=ap.parse_args();torch.set_num_threads(1);modes=['aligned_shared_base','independent_roles']
 if a.phase=='dev':worlds=[(25000,250000)];lrs=[.003,.01]
 else:
  if a.lr not in (.003,.01):raise SystemExit('fresh requires frozen lr')
  worlds=[(25001,250001),(25002,250002),(25003,250003)];lrs=[a.lr]
 rows=[];scores={lr:[] for lr in lrs}
 for wid,init in worlds:
  for mode in modes:
   w=make_world(wid,mode);data_seed=init+(0 if mode==modes[0] else 10000)
   for lr in lrs:
    for i,method in enumerate(METHODS):
     m,elapsed=fit(method,w,mode,init+i*191+int(lr*10000),data_seed,lr);mse,r2,x,role=evaluate(m,w,mode,init+91)
     if a.phase=='dev':scores[lr].append(mse)
     row={'condition':a.phase,'world_or_seed':wid,'mode':mode,'method':method,'serialized_model_bytes':m.serialized_payload_bytes(),'train_examples':UPDATES*64,'optimizer_updates':UPDATES,'active_compute_proxy':compute_proxy(method,UPDATES*64),'wall_time_s':round(elapsed,6),'inference_examples_per_s':round(throughput(m,x,role),3),'primary_metric':'heldout_MSE','primary_value':f'{mse:.10g}','secondary_metric':'R2','secondary_value':f'{r2:.8g}','status_note':f'lr={lr}; shared minibatches; address seed+algorithm charged'};rows.append(row);print(a.phase,wid,mode,method,lr,'MSE',mse,'R2',r2,'bytes',row['serialized_model_bytes'],flush=True)
 if a.phase=='dev':
  best=min(scores,key=lambda lr:sum(scores[lr])/len(scores[lr]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-250','selected_common_lr':best,'mean_mse_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'fresh_worlds':[25001,25002,25003],'updates':UPDATES},indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writerows(rows)
 print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),'rows':len(rows)}))
if __name__=='__main__':main()
