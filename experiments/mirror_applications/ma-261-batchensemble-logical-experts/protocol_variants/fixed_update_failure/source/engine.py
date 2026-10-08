from __future__ import annotations
import argparse,csv,json,platform,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import Experts,METHODS,E,D,O,mac_proxy
ROOT=Path(__file__).resolve().parents[1];UPDATES=1200;BATCH=128;FIELDS=['condition','world_or_seed','method','serialized_bytes','train_examples','optimizer_updates','active_compute_proxy','wall_time_s','inference_examples_per_s','mean_mse','worst_expert_mse','expert0_mse','expert1_mse','status_note']
def world(seed):
 g=torch.Generator().manual_seed(seed);bias=torch.randn(O,generator=g)*.05;tw=[]
 for e in range(E):
  a=torch.zeros(D,O);a[e*8:(e+1)*8]=torch.randn(8,O,generator=torch.Generator().manual_seed(seed+100+e))*.7;tw.append(a)
 return torch.stack(tw),bias
def batch_data(seed,n):
 g=torch.Generator().manual_seed(seed);x=torch.randn(n,D,generator=g);e=(x[:,8:].square().sum(-1)>x[:,:8].square().sum(-1)).long();return x,e
def fit(method,teacher,bias,init_seed,data_seed,lr):
 torch.manual_seed(init_seed);m=Experts(method,router_seed=0);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);x,e=batch_data(data_seed,8192);y=torch.einsum('bd,bdo->bo',x,teacher[e])+bias;g=torch.Generator().manual_seed(data_seed+41);t=time.perf_counter();m.train()
 for _ in range(UPDATES):
  ix=torch.randint(len(x),(BATCH,),generator=g);loss=F.mse_loss(m(x[ix],e[ix]),y[ix]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m,time.perf_counter()-t
def evaluate(m,teacher,bias,seed):
 x,e=batch_data(seed,8192);y=torch.einsum('bd,bdo->bo',x,teacher[e])+bias;m.eval();vals=[]
 with torch.no_grad():
  for k in range(E):
   z=e==k;vals.append(float((m(x[z],e[z])-y[z]).square().mean()))
 return sum(vals)/E,max(vals),vals
def throughput(m):
 x,e=batch_data(1,256);m.eval()
 with torch.no_grad():
  for _ in range(5):m(x,e)
  t=time.perf_counter()
  for _ in range(100):m(x,e)
 return len(x)*100/(time.perf_counter()-t)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);ap.add_argument('--lr',type=float);a=ap.parse_args();torch.set_num_threads(1)
 if a.phase=='dev':worlds=[(26100,261000),(26101,261001)];lrs=[.003,.01]
 else:
  if a.lr not in (.003,.01):raise SystemExit('fresh requires frozen LR')
  worlds=[(26102,261002),(26103,261003),(26104,261004)];lrs=[a.lr]
 rows=[];scores={lr:[] for lr in lrs}
 for wid,init in worlds:
  teacher,bias=world(wid);train_seed=wid+41000;eval_seed=wid+49000
  for lr in lrs:
   for i,method in enumerate(METHODS):
    m,elapsed=fit(method,teacher,bias,init+i*131+int(lr*10000),train_seed,lr);mean,worst,per=evaluate(m,teacher,bias,eval_seed)
    if a.phase=='dev':scores[lr].append(mean)
    row={'condition':a.phase,'world_or_seed':wid,'method':method,'serialized_bytes':m.payload_bytes(),'train_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':mac_proxy(method,UPDATES*BATCH),'wall_time_s':round(elapsed,6),'inference_examples_per_s':round(throughput(m),3),'mean_mse':f'{mean:.10g}','worst_expert_mse':f'{worst:.10g}','expert0_mse':f'{per[0]:.10g}','expert1_mse':f'{per[1]:.10g}','status_note':f'lr={lr}; common train/eval examples within world; oracle router provided'};rows.append(row);print(a.phase,wid,method,lr,'mse',mean,'worst',worst,'bytes',row['serialized_bytes'],flush=True)
 if a.phase=='dev':
  best=min(scores,key=lambda k:sum(scores[k])/len(scores[k]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-261','selected_lr':best,'mean_mse_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'fresh_worlds':[26102,26103,26104]},indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writerows(rows)
 print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),'rows':len(rows)}))
if __name__=='__main__':main()
