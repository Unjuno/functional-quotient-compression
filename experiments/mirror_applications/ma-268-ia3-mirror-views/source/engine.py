from __future__ import annotations
import argparse,csv,json,platform,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import TaskNet,METHODS,T,D,H,O,rotate4,mac_proxy
ROOT=Path(__file__).resolve().parents[1];UPDATES=1000;BATCH=128;FIELDS=['condition','world_or_seed','method','serialized_bytes','train_examples','optimizer_updates','active_compute_proxy','wall_time_s','inference_examples_per_s','mean_task_mse','worst_task_mse','status_note']
def world(seed):
 g=torch.Generator().manual_seed(seed);w1=torch.randn(D,H,generator=g)*.35;b1=torch.randn(H,generator=g)*.1;w2=torch.randn(H,O,generator=g)*.4;b2=torch.randn(O,generator=g)*.1;angles=torch.randn(T,4,generator=g)*.55;targets=[]
 for t in range(T):
  x=torch.randn(1024,D,generator=torch.Generator().manual_seed(seed+200+t));h=torch.tanh(x@w1+b1);y=rotate4(h,angles[t])@w2+b2
  targets.append((x,y))
 return (w1,b1,w2,b2,angles),targets
def data(seed,n):
 g=torch.Generator().manual_seed(seed);return torch.randn(n,D,generator=g),torch.randint(T,(n,),generator=g)
def fit(method,teacher,targets,init,data_seed,lr):
 torch.manual_seed(init);m=TaskNet(method,init);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);x,t=data(data_seed,8192);ys=[]
 for k in range(T):ys.append(torch.tanh(x@teacher[0]+teacher[1]) if False else None)
 # teacher mapping evaluated for each task on same x
 with torch.no_grad():
  h=torch.tanh(x@teacher[0]+teacher[1]);ys=torch.stack([rotate4(h,teacher[4][k])@teacher[2]+teacher[3] for k in range(T)])
 g=torch.Generator().manual_seed(data_seed+17);start=time.perf_counter();m.train()
 for _ in range(UPDATES):
  ix=torch.randint(len(x),(BATCH,),generator=g);loss=F.mse_loss(m(x[ix],t[ix]),ys[t[ix],ix]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m,time.perf_counter()-start
def evaluate(m,teacher,seed):
 x,t=data(seed,8192);h=torch.tanh(x@teacher[0]+teacher[1]);vals=[];m.eval()
 with torch.no_grad():
  for k in range(T):
   y=rotate4(h,teacher[4][k])@teacher[2]+teacher[3];tt=torch.full((len(x),),k,dtype=torch.long);vals.append(float((m(x,tt)-y).square().mean()))
 return sum(vals)/T,max(vals)
def throughput(m):
 x,t=data(99,256);m.eval()
 with torch.no_grad():
  for _ in range(5):m(x,t)
  start=time.perf_counter()
  for _ in range(100):m(x,t)
 return len(x)*100/(time.perf_counter()-start)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);ap.add_argument('--lr',type=float);a=ap.parse_args();torch.set_num_threads(1)
 if a.phase=='dev':worlds=[(26800,268000),(26801,268001)];lrs=[.003,.01]
 else:
  if a.lr not in (.003,.01):raise SystemExit('fresh requires frozen LR')
  worlds=[(26802,268002),(26803,268003),(26804,268004)];lrs=[a.lr]
 rows=[];scores={lr:[] for lr in lrs}
 for wid,init in worlds:
  teacher,_=world(wid);data_seed=wid+41000;eval_seed=wid+49000
  for lr in lrs:
   for i,method in enumerate(METHODS):
    m,elapsed=fit(method,teacher,None,init+i*139+int(lr*10000),data_seed,lr);mean,worst=evaluate(m,teacher,eval_seed)
    if a.phase=='dev':scores[lr].append(mean)
    row={'condition':a.phase,'world_or_seed':wid,'method':method,'serialized_bytes':m.payload_bytes(),'train_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':mac_proxy(method,UPDATES*BATCH),'wall_time_s':round(elapsed,6),'inference_examples_per_s':round(throughput(m),3),'mean_task_mse':f'{mean:.10g}','worst_task_mse':f'{worst:.10g}','status_note':f'lr={lr}; same teacher/data for all methods within world'};rows.append(row);print(a.phase,wid,method,lr,'mse',mean,'worst',worst,'bytes',row['serialized_bytes'],flush=True)
 if a.phase=='dev':
  best=min(scores,key=lambda k:sum(scores[k])/len(scores[k]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-268','selected_lr':best,'mean_mse_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'fresh_worlds':[26802,26803,26804]},indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writerows(rows)
 print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),'rows':len(rows)}))
if __name__=='__main__':main()
