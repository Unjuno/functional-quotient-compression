"""Development and fresh runner for MA-011."""
import argparse,csv,io,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import DIM,ROLES,METHODS,ProductExperts,rotate

UPDATES,BATCH=1800,128

def teacher(mode,seed):
 g=torch.Generator().manual_seed(seed)
 if mode=='aligned':
  base=torch.randn(2,DIM,DIM,generator=g)*0.10+torch.eye(DIM).unsqueeze(0)*0.12
  ai=torch.rand(2,2,generator=g)*1.2-0.6;ao=torch.rand(2,2,generator=g)*1.2-0.6
  codes=([0,0,1,1],[0,1,0,1]);f=[]
  for b in range(2):
   f.append(torch.stack([rotate(torch.tensor(float(ao[b,codes[b][r]]))) @ base[b] @ rotate(torch.tensor(float(ai[b,codes[b][r]])),True) for r in range(ROLES)]))
  factors=torch.stack(f)
 elif mode=='independent':
  factors=torch.randn(2,ROLES,DIM,DIM,generator=g)*0.10+torch.eye(DIM).view(1,1,DIM,DIM)*0.12
 else: raise ValueError(mode)
 return factors

def sample(factors,seed,n):
 g=torch.Generator().manual_seed(seed);x=torch.randn(n,DIM,generator=g);role=torch.randint(ROLES,(n,),generator=g)
 return x,role,target(factors,x,role)

def target(factors,x,role):
 a=role//2;b=role%2
 ya=torch.tanh(torch.bmm(factors[0,a],x.unsqueeze(-1)).squeeze(-1))
 yb=torch.tanh(torch.bmm(factors[1,b],x.unsqueeze(-1)).squeeze(-1))
 return ya*yb

def bytes_for(m,method):
 b=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},b)
 cfg=json.dumps({'method':method,'dim':DIM,'roles':ROLES,'factor_count':2},sort_keys=True,separators=(',',':')).encode()
 return len(b.getvalue())+len(cfg)

def mac(method):
 base=2*DIM*DIM
 if method=='rank2':return base+2*2*DIM*2
 return base

def run_one(mode,world,tseed,iseed,method,lr):
 t=teacher(mode,tseed);x,r,y=sample(t,tseed+1,UPDATES*BATCH);xe,re,ye=sample(t,tseed+2,8192)
 m=ProductExperts(method,iseed);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
 gen=torch.Generator().manual_seed(world+100000);idx=torch.randint(len(x),(UPDATES,BATCH),generator=gen)
 st=time.perf_counter()
 for i in range(UPDATES):
  q=idx[i];opt.zero_grad(set_to_none=True);loss=F.mse_loss(m(x[q],r[q]),y[q]);loss.backward();opt.step()
 wall=time.perf_counter()-st
 with torch.no_grad():
  err=(m(xe,re)-ye).square().mean(1)
  roles_mse={str(k):float(err[re==k].mean()) for k in range(ROLES)}
  predscale=float(m(xe,re).abs().mean())
  reps=20;st=time.perf_counter()
  for _ in range(reps):m(xe[:1024],re[:1024])
  speed=1024/((time.perf_counter()-st)/reps)
 return {'mode':mode,'world':world,'teacher_seed':tseed,'initialization_seed':iseed,'method':method,'mse':float(err.mean()),'max_role_mse':max(roles_mse.values()),'role_mse':roles_mse,'output_abs_mean':predscale,'bytes':bytes_for(m,method),'examples':UPDATES*BATCH,'updates':UPDATES,'active_mac_proxy':mac(method),'train_wall_seconds':wall,'inference_examples_per_second':speed}

def save(path,rows):
 keys=['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note']
 p=Path(path);exists=p.exists()
 with p.open('a' if exists else 'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n')
  if not exists:w.writeheader()
  for x in rows:
   note={k:x[k] for k in ['teacher_seed','initialization_seed','lr','role_mse','output_abs_mean','inference_examples_per_second'] if k in x}
   w.writerow({'condition':x['mode'],'world_or_seed':x['world'],'method':x['method'],'serialized_bytes':x['bytes'],'train_tokens_or_examples':x['examples'],'optimizer_updates':x['updates'],'active_compute_proxy':x['active_mac_proxy'],'wall_time_s':x['train_wall_seconds'],'primary_metric':'heldout_product_mse','primary_value':x['mse'],'secondary_metric':'max_role_mse','secondary_value':x['max_role_mse'],'status_note':json.dumps(note,sort_keys=True)})

def development(out):
 allr=[];lrs=[.001,.003,.01]
 for lr in lrs:
  for mode in ('aligned','independent'):
   for i,m in enumerate(METHODS):
    x=run_one(mode,110000,1100000+(mode=='independent')*100,11000000+i*1000,m,lr);x.update(lr=lr);allr.append(x)
 by={lr:[x['mse'] for x in allr if x['lr']==lr] for lr in lrs};selected=min(lrs,key=lambda lr:sum(by[lr])/len(by[lr]))
 Path(out).mkdir(parents=True,exist_ok=True);save(Path(out)/'RESULTS_CORE.csv',allr)
 Path(out,'dev_raw.json').write_text(json.dumps(allr,indent=2,sort_keys=True)+'\n')
 Path(out,'DEV_SELECTION.json').write_text(json.dumps({'selected_learning_rate':selected,'mean_heldout_mse_by_lr':{str(k):sum(v)/len(v) for k,v in by.items()},'development_world':110000,'fresh_accessed':False},indent=2,sort_keys=True)+'\n')
 print(json.dumps({'selected_learning_rate':selected,'means':{str(k):sum(v)/len(v) for k,v in by.items()}},sort_keys=True))

def fresh(out,lr):
 rows=[]
 for w,t,i in zip((110001,110002,110003),(1100001,1100002,1100003),(11000001,11000002,11000003)):
  for mode in ('aligned','independent'):
   for j,m in enumerate(METHODS):
    x=run_one(mode,w,t+(mode=='independent')*100,i+j*1000,m,lr);x['lr']=lr;rows.append(x)
 save(Path(out)/'RESULTS_CORE.csv',rows);Path(out,'fresh_raw.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'fresh_rows':len(rows),'aligned_mirror_to_independent_mse':{str(w):next(x['mse'] for x in rows if x['world']==w and x['mode']=='aligned' and x['method']=='mirror')/next(x['mse'] for x in rows if x['world']==w and x['mode']=='aligned' and x['method']=='independent') for w in (110001,110002,110003)}}))

def main():
 torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('phase',choices=['development','fresh']);p.add_argument('--out',required=True);p.add_argument('--lr',type=float);a=p.parse_args()
 if a.phase=='development':development(a.out)
 else:
  if a.lr is None:raise SystemExit('--lr required')
  fresh(a.out,a.lr)
if __name__=='__main__':main()
