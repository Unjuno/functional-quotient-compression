#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import METHODS,MACS,InterventionModel

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads'
D,C,R=64,8,4;UPDATES=1000;BATCH=128;DEV=[50400,50401];FRESH=[50410,50411,50412];SEEDS=[0,1,2];LRS=[.003,.01]
FIELDS=['phase','world','seed','learning_rate','method','train_tokens','optimizer_updates','train_wall_seconds','mac_proxy_per_token','inference_tokens_per_second','intervention_nrmse','output_nrmse','serialized_payload_bytes','payload_sha256','roundtrip_max_abs_error','payload_path']
torch.set_num_threads(1)

def seed_all(s): random.seed(s);np.random.seed(s);torch.manual_seed(s)
def make_world(world,seed):
 g=torch.Generator().manual_seed(world*100003+seed*7919+504)
 teacher=torch.Generator().manual_seed(world*100003+seed*7919+905)
 basis=torch.linalg.qr(torch.randn(D,R,generator=teacher)).Q.contiguous()
 proj=torch.randn(R,D,generator=teacher)*(.9/(D**.5))
 angle=torch.randn(2,C,generator=teacher)*.35
 data={}
 for name,n,salt in [('train',8192,11),('validation',2048,23),('test',4096,37)]:
  gg=torch.Generator().manual_seed(world+seed*1009+salt)
  h=torch.randn(n,D,generator=gg);u=torch.randn(n,C,generator=gg)
  q=h@proj.T;ang=u@angle.T
  from model import rotate
  delta=rotate(q,ang)@basis.T;y=h+delta
  data[name]=(h,u,delta,y)
 return basis,data

def pack(method,model,world,seed,lr):
 obj={'method':method,'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'meta':{'world':world,'seed':seed,'lr':lr,'d':D,'context_dim':C,'rank':R}}
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def unpack(payload):
 o=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
 basis=o['state']['basis'] if 'basis' in o['state'] else torch.linalg.qr(torch.eye(D,R)).Q
 m=InterventionModel(o['method'],basis);m.load_state_dict(o['state']);m.eval();return m,o

def train(method,basis,data,seed,lr):
 seed_all(seed+METHODS.index(method)*43)
 model=InterventionModel(method,basis)
 h,u,d,_=data['train'];g=torch.Generator().manual_seed(seed+119)
 if method == 'shared':
  return model.eval(), 0.0
 opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
 start=time.perf_counter();model.train()
 for _ in range(UPDATES):
  ids=torch.randint(len(h),(BATCH,),generator=g)
  pred=model(h[ids],u[ids]);loss=F.mse_loss(pred,d[ids]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return model.eval(),time.perf_counter()-start

def evaluate(method,payload,data,phase,world,seed,lr,train_s):
 model,_=unpack(payload);h,u,target,y=data['test']
 with torch.no_grad():
  pd=model(h,u);inter=float(torch.linalg.norm(pd-target)/(torch.linalg.norm(target)+1e-12));out=float(torch.linalg.norm((h+pd)-y)/(torch.linalg.norm(y)+1e-12))
  xh,xu=h[:256],u[:256]
  for _ in range(5):model(xh,xu)
  t=time.perf_counter()
  for _ in range(30):model(xh,xu)
  tps=30*len(xh)/max(time.perf_counter()-t,1e-9)
  # Exact deterministic output replay from serialized state.
  fresh,_=unpack(payload);rd=float((fresh(xh,xu)-model(xh,xu)).abs().max())
 digest=hashlib.sha256(payload).hexdigest();path=PAY/phase/str(world)/str(seed)/(method+'.pt');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
 return {'phase':phase,'world':world,'seed':seed,'learning_rate':lr,'method':method,'train_tokens':(0 if method=='shared' else UPDATES*BATCH),'optimizer_updates':(0 if method=='shared' else UPDATES),'train_wall_seconds':round(train_s,6),'mac_proxy_per_token':MACS[method],'inference_tokens_per_second':round(tps,3),'intervention_nrmse':f'{inter:.9f}','output_nrmse':f'{out:.9f}','serialized_payload_bytes':len(payload),'payload_sha256':digest,'roundtrip_max_abs_error':rd,'payload_path':str(path.relative_to(ROOT.parents[2]))}

def run(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True);rows=[]
 if phase=='development': worlds=DEV;lrs=LRS
 else:
  sel=ART/'DEV_SELECTION.json'
  if not sel.exists():raise SystemExit('development selection missing')
  worlds=FRESH;lrs=[float(json.loads(sel.read_text())['selected_learning_rate'])]
 score={lr:[] for lr in lrs}
 for world in worlds:
  for seed in SEEDS:
   basis,data=make_world(world,seed)
   for lr in lrs:
    for method in METHODS:
     model,sec=train(method,basis,data,world*100+seed,lr);payload=pack(method,model,world,seed,lr)
     row=evaluate(method,payload,data,phase,world,seed,lr,sec);rows.append(row)
     if phase=='development':score[lr].append(float(row['intervention_nrmse']))
     print(json.dumps({'phase':phase,'world':world,'seed':seed,'lr':lr,'method':method,'rows':len(rows)}),flush=True)
 dest=ART/('development.csv' if phase=='development' else 'fresh.csv')
 with dest.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n');w.writeheader();w.writerows(rows)
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n');w.writeheader();w.writerows(rows)
 (ART/('development_runs.jsonl' if phase=='development' else 'fresh_runs.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
 if phase=='development':
  means={str(lr):sum(v)/len(v) for lr,v in score.items()};chosen=min(lrs,key=lambda lr:means[str(lr)])
  (ART/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-504','selected_learning_rate':chosen,'mean_validation_intervention_nrmse':means,'fresh_worlds_locked':FRESH,'seeds':SEEDS},indent=2)+'\n')
 print(json.dumps({'phase':phase,'rows':len(rows),'selected_lr':(json.loads((ART/'DEV_SELECTION.json').read_text())['selected_learning_rate'] if phase=='development' else lrs[0])},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);a=p.parse_args();run(a.phase)
