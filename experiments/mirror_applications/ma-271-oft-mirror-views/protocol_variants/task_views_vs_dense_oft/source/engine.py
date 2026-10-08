from __future__ import annotations
import argparse,csv,json,platform,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import TaskViews,METHODS,T,D,O,givens,mac_proxy
ROOT=Path(__file__).resolve().parents[1];UPDATES=1000;BATCH=128;FIELDS=['condition','world_or_seed','method','serialized_bytes','train_examples','optimizer_updates','active_compute_proxy','wall_time_s','inference_examples_per_s','mean_mse','worst_task_mse','status_note']
def world(seed):
 g=torch.Generator().manual_seed(seed);base=torch.randn(D,O,generator=g)*.5;angs=torch.randn(T,8,generator=g)*.4;targets=[]
 for t in range(T):
  x=torch.eye(O);targets.append(base@givens(x,angs[t]))
 return base,angs,torch.stack(targets)
def data(seed,n):
 g=torch.Generator().manual_seed(seed);return torch.randn(n,D,generator=g),torch.randint(T,(n,),generator=g)
def fit(method,teacher,targets,init,data_seed,lr):
 torch.manual_seed(init);m=TaskViews(method,init);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);x,t=data(data_seed,8192);y=torch.einsum('bd,bdo->bo',x,targets[t]);g=torch.Generator().manual_seed(data_seed+19);start=time.perf_counter();m.train()
 for _ in range(UPDATES):
  ix=torch.randint(len(x),(BATCH,),generator=g);loss=F.mse_loss(m(x[ix],t[ix]),y[ix]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m,time.perf_counter()-start
def evaluate(m,targets,seed):
 x,t=data(seed,8192);y=torch.einsum('bd,bdo->bo',x,targets[t]);m.eval();vals=[]
 with torch.no_grad():
  for k in range(T):
   q=t==k;tt=torch.full((int(q.sum()),),k,dtype=torch.long);vals.append(float((m(x[q],tt)-y[q]).square().mean()))
 return sum(vals)/T,max(vals)
def throughput(m):
 x,t=data(7,128);m.eval()
 with torch.no_grad():
  for _ in range(5):m(x,t)
  start=time.perf_counter()
  for _ in range(50):m(x,t)
 return len(x)*50/(time.perf_counter()-start)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);ap.add_argument('--lr',type=float);a=ap.parse_args();torch.set_num_threads(1)
 if a.phase=='dev':worlds=[(27100,271000),(27101,271001)];lrs=[.003,.01]
 else:
  if a.lr not in (.003,.01):raise SystemExit('fresh requires frozen LR')
  worlds=[(27102,271002),(27103,271003),(27104,271004)];lrs=[a.lr]
 rows=[];scores={lr:[] for lr in lrs}
 for wid,init in worlds:
  _,_,targets=world(wid);ds=wid+41000;ev=wid+49000
  for lr in lrs:
   for i,method in enumerate(METHODS):
    m,elapsed=fit(method,None,targets,init+i*149+int(lr*10000),ds,lr);mean,worst=evaluate(m,targets,ev)
    if a.phase=='dev':scores[lr].append(mean)
    row={'condition':a.phase,'world_or_seed':wid,'method':method,'serialized_bytes':m.payload_bytes(),'train_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':mac_proxy(method,UPDATES*BATCH),'wall_time_s':round(elapsed,6),'inference_examples_per_s':round(throughput(m),3),'mean_mse':f'{mean:.10g}','worst_task_mse':f'{worst:.10g}','status_note':f'lr={lr}; common teacher/data per world'};rows.append(row);print(a.phase,wid,method,lr,'mse',mean,'worst',worst,'bytes',row['serialized_bytes'],flush=True)
 if a.phase=='dev':
  best=min(scores,key=lambda k:sum(scores[k])/len(scores[k]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-271','selected_lr':best,'mean_mse_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'fresh_worlds':[27102,27103,27104]},indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writerows(rows)
 print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),'rows':len(rows)}))
if __name__=='__main__':main()
