import argparse,csv,io,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import DIN,HID,DOUT,ROLES,METHODS,MultiplicativeExperts
UPDATES,BATCH=1200,64

def teacher(mode,seed):
 g=torch.Generator().manual_seed(seed)
 if mode=='aligned':return {'mode':mode,'wi':torch.randn(HID,DIN,generator=g)*.12,'wo':torch.randn(DOUT,HID,generator=g)*.12,'u':torch.randn(HID,generator=g)*.2,'v':torch.randn(HID,generator=g)*.2,'alpha':torch.randn(2,generator=g)*.6,'beta':torch.randn(2,generator=g)*.6}
 return {'mode':mode,'wi':torch.randn(ROLES,HID,DIN,generator=g)*.12,'wo':torch.randn(ROLES,DOUT,HID,generator=g)*.12}

def target(t,x,r):
 if t['mode']=='aligned':
  h=F.gelu(torch.bmm(t['wi'].expand(len(x),-1,-1),x.unsqueeze(-1)).squeeze(-1));a=t['alpha'][r//2,None];b=t['beta'][r%2,None];h=h*((1+a*t['u'])*(1+b*t['v']));return torch.bmm(t['wo'].expand(len(x),-1,-1),h.unsqueeze(-1)).squeeze(-1)
 h=F.gelu(torch.bmm(t['wi'][r],x.unsqueeze(-1)).squeeze(-1));return torch.bmm(t['wo'][r],h.unsqueeze(-1)).squeeze(-1)
def data(t,seed,n):
 g=torch.Generator().manual_seed(seed);x=torch.randn(n,DIN,generator=g);r=torch.arange(n)%ROLES;p=torch.randperm(n,generator=g);x=x[p];r=r[p];return x,r,target(t,x,r)
def payload(m,method):
 b=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},b);cfg=json.dumps({'method':method,'din':DIN,'hidden':HID,'dout':DOUT,'roles':ROLES},sort_keys=True,separators=(',',':')).encode();return len(b.getvalue())+len(cfg)
def mac(method):
 n=2*DIN*HID
 return n+(2*(DOUT+HID) if method=='rank2' else 0)
def run(mode,world,ts,is_,method,lr):
 t=teacher(mode,ts);x,r,y=data(t,ts+1,UPDATES*BATCH);xe,re,ye=data(t,ts+2,4096);m=MultiplicativeExperts(method,is_);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(world+130000);ix=torch.randint(len(x),(UPDATES,BATCH),generator=g);st=time.perf_counter()
 for i in range(UPDATES):
  q=ix[i];opt.zero_grad(set_to_none=True);loss=F.mse_loss(m(x[q],r[q]),y[q]);loss.backward();opt.step()
 wall=time.perf_counter()-st
 with torch.no_grad():
  e=(m(xe,re)-ye).square().mean(1);by={str(k):float(e[re==k].mean()) for k in range(ROLES)};reps=20;st=time.perf_counter()
  for _ in range(reps):m(xe[:1024],re[:1024])
  speed=1024/((time.perf_counter()-st)/reps)
 return {'mode':mode,'world':world,'teacher_seed':ts,'initialization_seed':is_,'method':method,'lr':lr,'mse':float(e.mean()),'max_role_mse':max(by.values()),'role_mse':by,'bytes':payload(m,method),'examples':UPDATES*BATCH,'updates':UPDATES,'active_mac_proxy':mac(method),'train_wall_seconds':wall,'inference_examples_per_second':speed}
def save(path,rows):
 keys=['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'];p=Path(path);new=not p.exists()
 with p.open('a' if not new else 'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n')
  if new:w.writeheader()
  for r in rows:
   note={k:r[k] for k in ('teacher_seed','initialization_seed','lr','role_mse','inference_examples_per_second')};w.writerow({'condition':r['mode'],'world_or_seed':r['world'],'method':r['method'],'serialized_bytes':r['bytes'],'train_tokens_or_examples':r['examples'],'optimizer_updates':r['updates'],'active_compute_proxy':r['active_mac_proxy'],'wall_time_s':r['train_wall_seconds'],'primary_metric':'heldout_routed_mse','primary_value':r['mse'],'secondary_metric':'max_role_mse','secondary_value':r['max_role_mse'],'status_note':json.dumps(note,sort_keys=True)})
def development(out):
 rows=[];lrs=(.001,.003,.01)
 for lr in lrs:
  for mode in ('aligned','independent'):
   for j,m in enumerate(METHODS):rows.append(run(mode,130000,1300000+(mode=='independent')*100,13000000+j*1000,m,lr))
 means={lr:[r['mse'] for r in rows if r['lr']==lr] for lr in lrs};selected=min(lrs,key=lambda k:sum(means[k])/len(means[k]));Path(out).mkdir(parents=True,exist_ok=True);save(Path(out)/'RESULTS_CORE.csv',rows);Path(out,'dev_raw.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n');Path(out,'DEV_SELECTION.json').write_text(json.dumps({'selected_learning_rate':selected,'mean_mse_by_lr':{str(k):sum(v)/len(v) for k,v in means.items()},'development_world':130000,'fresh_accessed':False},indent=2,sort_keys=True)+'\n');print(json.dumps({'selected_lr':selected,'means':{str(k):sum(v)/len(v) for k,v in means.items()}},sort_keys=True))
def fresh(out,lr):
 rows=[]
 for w,ts,is_ in zip((130001,130002,130003),(1300001,1300002,1300003),(13000001,13000002,13000003)):
  for mode in ('aligned','independent'):
   for j,m in enumerate(METHODS):rows.append(run(mode,w,ts+(mode=='independent')*100,is_+j*1000,m,lr))
 Path(out).mkdir(parents=True,exist_ok=True);save(Path(out)/'RESULTS_CORE.csv',rows);Path(out,'fresh_raw.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'fresh_rows':len(rows),'aligned_mirror_to_full_mse':{str(w):next(q['mse'] for q in rows if q['world']==w and q['mode']=='aligned' and q['method']=='multiplicative')/next(q['mse'] for q in rows if q['world']==w and q['mode']=='aligned' and q['method']=='independent') for w in (130001,130002,130003)}},sort_keys=True))
def main():
 torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--lr',type=float);p.add_argument('phase',choices=['development','fresh']);a=p.parse_args()
 if a.phase=='development':development(a.out)
 else:
  if a.lr is None:raise SystemExit('--lr required')
  fresh(a.out,a.lr)
if __name__=='__main__':main()
