from __future__ import annotations
import argparse,csv,json,platform,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import RoutedFFN,METHODS,E,D,H,O,butterfly,mac_proxy
ROOT=Path(__file__).resolve().parents[1];UPDATES=1000;BATCH=128;FIELDS=['condition','world_or_seed','method','serialized_bytes','train_examples','optimizer_updates','active_compute_proxy','wall_time_s','inference_examples_per_s','mean_expert_mse','worst_expert_mse','expert0_mse','expert1_mse','expert2_mse','expert3_mse','status_note']
def world(seed):
 g=torch.Generator().manual_seed(seed);w1=torch.randn(D,H,generator=g)*.35;b1=torch.randn(H,generator=g)*.1;w2=torch.randn(H,O,generator=g)*.35;b2=torch.randn(O,generator=g)*.1;atom=torch.randn(4,8,generator=g)*.25;codes=torch.randn(E,generator=g)*.8;return w1,b1,w2,b2,atom,codes
def data(seed,n):
 g=torch.Generator().manual_seed(seed);x=torch.randn(n,D,generator=g);e=torch.randint(E,(n,),generator=g);return x,e
def targets(x,e,world):
 w1,b1,w2,b2,atom,codes=world;h=torch.tanh(x@w1+b1);out=[]
 for j in range(len(x)):out.append(butterfly(h[j:j+1],atom*codes[e[j]])@w2+b2)
 return torch.cat(out)
def fit(method,world,init,ds,lr):
 torch.manual_seed(init);m=RoutedFFN(method,init);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);x,e=data(ds,8192);y=targets(x,e,world);g=torch.Generator().manual_seed(ds+31);st=time.perf_counter();m.train()
 for _ in range(UPDATES):
  ix=torch.randint(len(x),(BATCH,),generator=g);loss=F.mse_loss(m(x[ix],e[ix]),y[ix]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m,time.perf_counter()-st
def evaluate(m,world,seed):
 x,e=data(seed,8192);y=targets(x,e,world);vals=[];m.eval()
 with torch.no_grad():
  for k in range(E):
   q=e==k;ek=torch.full((int(q.sum()),),k,dtype=torch.long);vals.append(float((m(x[q],ek)-y[q]).square().mean()))
 return sum(vals)/E,max(vals),vals
def throughput(m):
 x,e=data(7,256);m.eval()
 with torch.no_grad():
  for _ in range(3):m(x,e)
  st=time.perf_counter()
  for _ in range(30):m(x,e)
 return len(x)*30/(time.perf_counter()-st)
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['dev','fresh'],required=True);p.add_argument('--lr',type=float);a=p.parse_args();torch.set_num_threads(1)
 if a.phase=='dev':ws=[(27400,274000),(27401,274001)];lrs=[.003,.01]
 else:
  if a.lr not in (.003,.01):raise SystemExit('fresh requires frozen LR')
  ws=[(27402,274002),(27403,274003),(27404,274004)];lrs=[a.lr]
 rows=[];scores={lr:[] for lr in lrs}
 for wid,init in ws:
  w=world(wid);ds=wid+41000;ev=wid+49000
  for lr in lrs:
   for i,method in enumerate(METHODS):
    m,wall=fit(method,w,init+i*167+int(lr*10000),ds,lr);mean,worst,per=evaluate(m,w,ev)
    if a.phase=='dev':scores[lr].append(mean)
    row={'condition':a.phase,'world_or_seed':wid,'method':method,'serialized_bytes':m.payload_bytes(),'train_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':mac_proxy(method,UPDATES*BATCH),'wall_time_s':round(wall,6),'inference_examples_per_s':round(throughput(m),3),'mean_expert_mse':f'{mean:.10g}','worst_expert_mse':f'{worst:.10g}','expert0_mse':f'{per[0]:.10g}','expert1_mse':f'{per[1]:.10g}','expert2_mse':f'{per[2]:.10g}','expert3_mse':f'{per[3]:.10g}','status_note':f'lr={lr}; oracle routes; common data per world'};rows.append(row);print(a.phase,wid,method,lr,mean,worst,row['serialized_bytes'],flush=True)
 if a.phase=='dev':
  best=min(scores,key=lambda k:sum(scores[k])/len(scores[k]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'selected_lr':best,'mean_mse_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'fresh_worlds':[27402,27403,27404]},indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writerows(rows)
 print(json.dumps({'torch':torch.__version__,'python':platform.python_version(),'rows':len(rows)}))
if __name__=='__main__':main()
