from __future__ import annotations
import argparse,csv,json,platform,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import METHODS,T,D,O,TaskFamilyModel,mac_proxy
ROOT=Path(__file__).resolve().parents[1];FIELDS=['condition','world_or_seed','method','serialized_bytes','train_examples','optimizer_updates','active_compute_proxy','wall_time_s','inference_examples_per_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'];UPDATES=1200;BATCH=128

def world(seed):
 g=torch.Generator().manual_seed(seed);base=torch.randn(D,O,generator=g)*.35;ang=torch.randn(T,8,generator=g)*.45;gain=.6+torch.rand(T,O,generator=g)*.8;targets=[]
 for t in range(T):
  p=torch.eye(O)
  for i in range(8):
   a=ang[t,i];c,s=torch.cos(a),torch.sin(a);r=torch.eye(O);r[2*i,2*i]=c;r[2*i,2*i+1]=-s;r[2*i+1,2*i]=s;r[2*i+1,2*i+1]=c;p=r@p
  targets.append(base@p.T@torch.diag(gain[t]))
 return torch.stack(targets)

def fit(method,targets,init_seed,data_seed,lr):
 torch.manual_seed(init_seed);m=TaskFamilyModel(method,context_seed=init_seed+77);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(data_seed);t0=time.perf_counter();m.train()
 for _ in range(UPDATES):
  x=torch.randn(BATCH,D,generator=g);task=torch.randint(T,(BATCH,),generator=g);y=torch.einsum('bd,bdo->bo',x,targets[task]);loss=F.mse_loss(m(x,task),y);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m,time.perf_counter()-t0

def evaluate(m,targets,seed):
 g=torch.Generator().manual_seed(seed);x=torch.randn(2048,D,generator=g);vals=[];m.eval()
 with torch.no_grad():
  for t in range(T):
   task=torch.full((len(x),),t,dtype=torch.long);vals.append(float((m(x,task)-x@targets[t]).square().mean()))
 return sum(vals)/T,max(vals),vals

def throughput(m):
 x=torch.randn(128,D);task=torch.randint(T,(128,));m.eval()
 with torch.no_grad():
  for _ in range(5):m(x,task)
  t=time.perf_counter()
  for _ in range(100):m(x,task)
 return 12800/(time.perf_counter()-t)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);ap.add_argument('--lr',type=float);a=ap.parse_args();torch.set_num_threads(1)
 if a.phase=='dev':worlds=[(25500,255000)];lrs=[.003,.01]
 else:
  if a.lr not in (.003,.01):raise SystemExit('fresh requires frozen LR')
  worlds=[(25501,255001),(25502,255002),(25503,255003)];lrs=[a.lr]
 rows=[];scores={lr:[] for lr in lrs}
 for wid,init in worlds:
  targets=world(wid)
  for lr in lrs:
   for i,method in enumerate(METHODS):
    m,elapsed=fit(method,targets,init+i*191+int(lr*10000),init+313,lr);mean,worst,per=evaluate(m,targets,init+991)
    if a.phase=='dev':scores[lr].append(mean)
    row={'condition':a.phase,'world_or_seed':wid,'method':method,'serialized_bytes':m.payload_bytes(),'train_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':mac_proxy(method,UPDATES*BATCH),'wall_time_s':round(elapsed,6),'inference_examples_per_s':round(throughput(m),3),'primary_metric':'mean_task_heldout_MSE','primary_value':f'{mean:.10g}','secondary_metric':'worst_task_heldout_MSE','secondary_value':f'{worst:.10g}','status_note':f'lr={lr}; per_task={json.dumps(per)}; matched data/update budget'}
    rows.append(row);print(a.phase,wid,method,lr,'mean',mean,'worst',worst,'bytes',row['serialized_bytes'],flush=True)
 if a.phase=='dev':
  best=min(scores,key=lambda k:sum(scores[k])/len(scores[k]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-255','selected_common_lr':best,'mean_mse_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'fresh_worlds':[25501,25502,25503],'updates':UPDATES,'fixed_before_fresh':True},indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writerows(rows)
 print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),'rows':len(rows)}))
if __name__=='__main__':main()
