from __future__ import annotations
import argparse,csv,json,platform,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import Bank,METHODS,T,D,O,butterfly,mac
ROOT=Path(__file__).resolve().parents[1];UPDATES=1000;BATCH=128;FIELDS=['condition','world_or_seed','method','serialized_bytes','train_examples','optimizer_updates','active_compute_proxy','wall_time_s','inference_examples_per_s','mean_task_mse','worst_task_mse','status_note']
def world(seed):
 g=torch.Generator().manual_seed(seed);w=torch.randn(D,O,generator=g)*.5;atom=torch.randn(4,8,generator=g)*.25;codes=torch.randn(T,generator=g)*.9;ang=atom[None]*codes[:,None,None];targets=torch.stack([butterfly(torch.eye(D),ang[t])@w for t in range(T)]);return w,atom,codes,targets
def data(seed,n):
 g=torch.Generator().manual_seed(seed);return torch.randn(n,D,generator=g),torch.randint(T,(n,),generator=g)
def fit(method,teacher,atom,codes,targets,init,ds,lr):
 torch.manual_seed(init);m=Bank(method,init);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
 x,t=data(ds,8192);y=torch.einsum('bd,bdo->bo',x,targets[t]);g=torch.Generator().manual_seed(ds+31);st=time.perf_counter();m.train()
 for _ in range(UPDATES):
  ix=torch.randint(len(x),(BATCH,),generator=g);loss=F.mse_loss(m(x[ix],t[ix]),y[ix]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m,time.perf_counter()-st
def evaluate(m,target,seed):
 x,t=data(seed,8192);y=torch.einsum('bd,bdo->bo',x,target[t]);vals=[];m.eval()
 with torch.no_grad():
  for k in range(T):
   q=t==k;vals.append(float((m(x[q],torch.full((int(q.sum()),),k))-y[q]).square().mean()))
 return sum(vals)/T,max(vals)
def throughput(m):
 x,t=data(7,128);m.eval()
 with torch.no_grad():
  for _ in range(3):m(x,t)
  st=time.perf_counter()
  for _ in range(30):m(x,t)
 return 3840/(time.perf_counter()-st)
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['dev','fresh'],required=True);p.add_argument('--lr',type=float);a=p.parse_args();torch.set_num_threads(1)
 if a.phase=='dev':ws=[(27300,273000),(27301,273001)];lrs=[.003,.01]
 else:
  if a.lr not in (.003,.01):raise SystemExit('fresh requires frozen lr')
  ws=[(27302,273002),(27303,273003),(27304,273004)];lrs=[a.lr]
 rows=[];scores={l:[] for l in lrs}
 for wid,init in ws:
  w,atom,codes,target=world(wid)
  for lr in lrs:
   for i,method in enumerate(METHODS):
    m,wall=fit(method,w,atom,codes,target,init+i*163+int(lr*10000),wid+41000,lr);mean,worst=evaluate(m,target,wid+49000)
    if a.phase=='dev':scores[lr].append(mean)
    row={'condition':a.phase,'world_or_seed':wid,'method':method,'serialized_bytes':m.payload_bytes(),'train_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':mac(method,UPDATES*BATCH),'wall_time_s':round(wall,6),'inference_examples_per_s':round(throughput(m),3),'mean_task_mse':f'{mean:.10g}','worst_task_mse':f'{worst:.10g}','status_note':f'lr={lr}; common teacher and data per world'};rows.append(row);print(a.phase,wid,method,lr,mean,worst,row['serialized_bytes'],flush=True)
 if a.phase=='dev':
  best=min(scores,key=lambda l:sum(scores[l])/len(scores[l]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'selected_lr':best,'mean_mse_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'fresh_worlds':[27302,27303,27304]},indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writerows(rows)
 print(json.dumps({'torch':torch.__version__,'python':platform.python_version(),'rows':len(rows)}))
if __name__=='__main__':main()
