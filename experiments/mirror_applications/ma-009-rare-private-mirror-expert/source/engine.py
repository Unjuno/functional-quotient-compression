#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,platform,time
from pathlib import Path
import torch
from model import METHODS,RoutedExperts,compute_proxy,givens,role_of
ROOT=Path(__file__).resolve().parents[1];D,O,N,UPDATES,BATCH=16,12,4,1200,64
FIELDS=["condition","world_or_seed","mode","method","serialized_model_bytes","train_examples","optimizer_updates","active_compute_proxy","wall_time_s","inference_examples_per_s","primary_metric","primary_value","secondary_metric","secondary_value","status_note"]

def make_world(seed,mode):
 g=torch.Generator().manual_seed(seed)
 if mode=="shared_common_private_rare":return {"shared":torch.randn(D,O,generator=g)*.6,"angles":torch.rand(N,D//2,generator=g)*1.1-.55,"rare":torch.randn(D,O,generator=g)*.6,"weights":None}
 return {"shared":None,"angles":None,"rare":None,"weights":torch.randn(N,D,O,generator=g)*.6}

def draw_x(n,seed,skewed=True):
 x=torch.randn(n,D,generator=torch.Generator().manual_seed(seed))
 if skewed:x[:,:2]+=1.2
 return x

def target(x,world,mode):
 role=role_of(x)
 if mode=="independent_all":return torch.einsum("bd,bdo->bo",x,world["weights"][role])
 y=torch.empty(x.shape[0],O)
 rare=role==0
 if rare.any():y[rare]=x[rare]@world["rare"]
 for r in range(1,N):
  m=role==r
  if m.any():y[m]=givens(x[m],world["angles"][r].expand(int(m.sum()),-1))@world["shared"]
 return y

def fit(method,world,mode,init_seed,data_seed,lr):
 torch.manual_seed(init_seed);model=RoutedExperts(method,init_seed);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(data_seed+23);start=time.perf_counter();model.train()
 for _ in range(UPDATES):
  x=torch.randn(BATCH,D,generator=g);x[:,:2]+=1.2;y=target(x,world,mode);loss=(model(x)-y).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return model,time.perf_counter()-start

def evaluate(model,world,mode,seed):
 x=draw_x(16384,seed,True);y=target(x,world,mode);balanced=draw_x(8192,seed+13,False);yb=target(balanced,world,mode);model.eval()
 with torch.no_grad():
  pred=model(x);pb=model(balanced);mse=float((pred-y).square().mean());rare=float((pred[role_of(x)==0]-y[role_of(x)==0]).square().mean());bmse=float((pb-yb).square().mean())
 return mse,rare,bmse,x

def throughput(model,x):
 x=x[:64];model.eval()
 with torch.no_grad():
  for _ in range(4):model(x)
  t=time.perf_counter()
  for _ in range(20):model(x)
 return 64*20/(time.perf_counter()-t)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);ap.add_argument('--lr',type=float);args=ap.parse_args();torch.set_num_threads(1);modes=['shared_common_private_rare','independent_all']
 if args.phase=='dev':worlds,lrs=[(90000,900000)],[.003,.01]
 else:
  if args.lr not in (.003,.01):raise SystemExit('fresh requires frozen LR')
  worlds,lrs=[(90001,900001),(90002,900002),(90003,900003)],[args.lr]
 rows,scores=[],{lr:[] for lr in lrs}
 for wid,init in worlds:
  for mode in modes:
   world=make_world(wid,mode);ds=init+(0 if mode==modes[0] else 10000)
   for lr in lrs:
    for i,method in enumerate(METHODS):
     model,elapsed=fit(method,world,mode,init+i*193+int(lr*10000),ds,lr);mse,rare,bmse,x=evaluate(model,world,mode,init+91)
     if args.phase=='dev':scores[lr].append(mse)
     row={"condition":args.phase,"world_or_seed":wid,"mode":mode,"method":method,"serialized_model_bytes":model.serialized_payload_bytes(),"train_examples":UPDATES*BATCH,"optimizer_updates":UPDATES,"active_compute_proxy":compute_proxy(method,UPDATES*BATCH),"wall_time_s":round(elapsed,6),"inference_examples_per_s":round(throughput(model,x),3),"primary_metric":"natural_routed_MSE","primary_value":f"{mse:.10g}","secondary_metric":"rare_role_MSE;balanced_MSE","secondary_value":f"{rare:.10g};{bmse:.10g}","status_note":f"lr={lr}; matched minibatches; oracle hard route; rare role=0; shift=1.2"};rows.append(row)
     print(args.phase,wid,mode,method,lr,'MSE',mse,'rare',rare,'balanced',bmse,'bytes',row['serialized_model_bytes'],flush=True)
 if args.phase=='dev':
  best=min(scores,key=lambda lr:sum(scores[lr])/len(scores[lr]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-009','selected_common_lr':best,'mean_mse_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'fresh_worlds':[90001,90002,90003],'updates':UPDATES},indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writerows(rows)
 print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),'rows':len(rows)}))
if __name__=='__main__':main()
