from __future__ import annotations
import argparse,csv,json,platform,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import AdapterBank,METHODS,T,D,O,R,mac_proxy
ROOT=Path(__file__).resolve().parents[1];UPDATES=1000;BATCH=128;FIELDS=['condition','world_or_seed','method','serialized_bytes','train_examples','optimizer_updates','active_compute_proxy','wall_time_s','inference_examples_per_s','mean_task_mse','worst_task_mse','status_note']
def make_world(seed,basis_seed):
 g=torch.Generator().manual_seed(seed);a=torch.randn(D,R,generator=torch.Generator().manual_seed(basis_seed))/D**.5;b=torch.randn(R,O,generator=torch.Generator().manual_seed(basis_seed+1))/R**.5;coef=torch.randn(T,R,generator=g)*.7;targets=torch.stack([a@torch.diag(coef[t])@b for t in range(T)]);return targets

def data(seed,n):
 g=torch.Generator().manual_seed(seed);x=torch.randn(n,D,generator=g);t=torch.randint(T,(n,),generator=g);return x,t
def fit(method,targets,init,basis_seed,data_seed,lr):
 torch.manual_seed(init);m=AdapterBank(method,basis_seed);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);x,t=data(data_seed,8192);y=torch.einsum('bd,bdo->bo',x,targets[t]);g=torch.Generator().manual_seed(data_seed+17);start=time.perf_counter();m.train()
 for _ in range(UPDATES):
  ix=torch.randint(len(x),(BATCH,),generator=g);loss=F.mse_loss(m(x[ix],t[ix]),y[ix]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m,time.perf_counter()-start
def evaluate(m,targets,seed):
 x,t=data(seed,8192);y=torch.einsum('bd,bdo->bo',x,targets[t]);m.eval();vals=[]
 with torch.no_grad():
  for k in range(T):
   q=t==k;vals.append(float((m(x[q],t[q])-y[q]).square().mean()))
 return sum(vals)/T,max(vals)
def throughput(m):
 x,t=data(4,256);m.eval()
 with torch.no_grad():
  for _ in range(5):m(x,t)
  start=time.perf_counter()
  for _ in range(100):m(x,t)
 return len(x)*100/(time.perf_counter()-start)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);ap.add_argument('--lr',type=float);a=ap.parse_args();torch.set_num_threads(1)
 if a.phase=='dev':worlds=[(26500,265000),(26501,265001)];lrs=[.003,.01]
 else:
  if a.lr not in (.003,.01):raise SystemExit('fresh requires frozen lr')
  worlds=[(26502,265002),(26503,265003),(26504,265004)];lrs=[a.lr]
 rows=[];scores={lr:[] for lr in lrs}
 for wid,init in worlds:
  basis_seed=wid+30000;targets=make_world(wid,basis_seed);tr=wid+41000;ev=wid+49000
  for lr in lrs:
   for i,method in enumerate(METHODS):
    m,elapsed=fit(method,targets,init+i*137+int(lr*10000),basis_seed,tr,lr);mean,worst=evaluate(m,targets,ev)
    if a.phase=='dev':scores[lr].append(mean)
    row={'condition':a.phase,'world_or_seed':wid,'method':method,'serialized_bytes':m.payload_bytes(),'train_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':mac_proxy(method,UPDATES*BATCH),'wall_time_s':round(elapsed,6),'inference_examples_per_s':round(throughput(m),3),'mean_task_mse':f'{mean:.10g}','worst_task_mse':f'{worst:.10g}','status_note':f'lr={lr}; common basis, teacher and data within world'};rows.append(row);print(a.phase,wid,method,lr,'mse',mean,'worst',worst,'bytes',row['serialized_bytes'],flush=True)
 if a.phase=='dev':
  best=min(scores,key=lambda k:sum(scores[k])/len(scores[k]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-265','selected_lr':best,'mean_mse_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'fresh_worlds':[26502,26503,26504]},indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writerows(rows)
 print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),'rows':len(rows)}))
if __name__=='__main__':main()
