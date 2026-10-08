from __future__ import annotations
import argparse,csv,json,math,time,platform
from pathlib import Path
import torch
from torch.nn import functional as F
from model import Ensemble,METHODS,M,D,H,C,mac_proxy
ROOT=Path(__file__).resolve().parents[1];UPDATES=1000;BATCH=128;FIELDS=['condition','world_or_seed','method','serialized_bytes','train_examples','optimizer_updates','active_compute_proxy','wall_time_s','inference_examples_per_s','accuracy','nll','ece','pairwise_disagreement','status_note']

def data(seed,n):
 g=torch.Generator().manual_seed(seed);x=torch.randn(n,D,generator=g)
 proto=torch.randn(C,D,generator=torch.Generator().manual_seed(seed+991))*2.0
 # Balanced, learnable classes with explicit cluster separation.
 y=torch.arange(n)%C
 x=proto[y]+0.65*x
 perm=torch.randperm(n,generator=g)
 return x[perm],y[perm]

def fit(method,init_seed,data_seed,lr):
 torch.manual_seed(init_seed);m=Ensemble(method,init_seed);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);x,y=data(data_seed,8192);g=torch.Generator().manual_seed(data_seed+77);t0=time.perf_counter();m.train()
 for _ in range(UPDATES):
  ix=torch.randint(len(x),(BATCH,),generator=g);xx=x[ix];yy=y[ix];z=m.forward_members(xx);loss=torch.stack([F.cross_entropy(z[:,k],yy) for k in range(M)]).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m,time.perf_counter()-t0

def ece(probs,y,bins=10):
 conf,pred=probs.max(-1);acc=(pred==y).float();v=0.
 for i in range(bins):
  lo=i/bins;hi=(i+1)/bins;mask=(conf>=lo)&((conf<hi) if i<bins-1 else (conf<=hi))
  if mask.any():v+=mask.float().mean()*abs(float(acc[mask].mean()-conf[mask].mean()))
 return float(v)

def evaluate(m,seed):
 x,y=data(seed,4096);m.eval()
 with torch.no_grad():
  z=m.forward_members(x);lp=z.log_softmax(-1);probs=lp.exp();ens=probs.mean(1);pred=ens.argmax(-1);acc=float((pred==y).float().mean());nll=float(-ens[torch.arange(len(y)),y].clamp_min(1e-12).log().mean());cal=ece(ens,y)
  pa=probs.argmax(-1);d=[]
  for i in range(M):
   for j in range(i+1,M):d.append(float((pa[:,i]!=pa[:,j]).float().mean()))
 return acc,nll,cal,sum(d)/len(d)

def throughput(m):
 x=torch.randn(128,D);m.eval()
 with torch.no_grad():
  for _ in range(5):m.forward_members(x)
  t=time.perf_counter()
  for _ in range(80):m.forward_members(x)
 return 10240/(time.perf_counter()-t)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);ap.add_argument('--lr',type=float);a=ap.parse_args();torch.set_num_threads(1)
 if a.phase=='dev':worlds=[(26000,260000),(26001,260001)];lrs=[.003,.01]
 else:
  if a.lr not in (.003,.01):raise SystemExit('fresh requires frozen LR')
  worlds=[(26002,260002),(26003,260003),(26004,260004)];lrs=[a.lr]
 rows=[];scores={lr:[] for lr in lrs}
 for wid,init in worlds:
  for lr in lrs:
   for i,method in enumerate(METHODS):
    m,elapsed=fit(method,init+i*151+int(lr*10000),wid+41000,lr);acc,nll,cal,dis=evaluate(m,wid+49000)
    if a.phase=='dev':scores[lr].append(nll)
    row={'condition':a.phase,'world_or_seed':wid,'method':method,'serialized_bytes':m.payload_bytes(),'train_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':mac_proxy(method,UPDATES*BATCH),'wall_time_s':round(elapsed,6),'inference_examples_per_s':round(throughput(m),3),'accuracy':f'{acc:.8g}','nll':f'{nll:.10g}','ece':f'{cal:.8g}','pairwise_disagreement':f'{dis:.8g}','status_note':f'lr={lr}; same update/example budget'};rows.append(row);print(a.phase,wid,method,lr,'acc',acc,'nll',nll,'ece',cal,'dis',dis,'bytes',row['serialized_bytes'],flush=True)
 if a.phase=='dev':
  best=min(scores,key=lambda k:sum(scores[k])/len(scores[k]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-260','selected_lr':best,'mean_nll_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'fresh_worlds':[26002,26003,26004]},indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writerows(rows)
 print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),'rows':len(rows)}))
if __name__=='__main__':main()
