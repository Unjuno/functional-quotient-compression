import argparse,csv,io,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import DIN,HID,DOUT,DOMAINS,EXPERTS,METHODS,DomainExperts,rotate
UPDATES,BATCH=1200,128
def teacher(mode,seed):
 g=torch.Generator().manual_seed(seed)
 if mode=='aligned':return {'mode':mode,'wi':torch.randn(EXPERTS,HID,DIN,generator=g)*.12,'wo':torch.randn(EXPERTS,DOUT,HID,generator=g)*.12,'ai':torch.rand(DOMAINS,generator=g)*1.2-.6,'ao':torch.rand(DOMAINS,generator=g)*1.2-.6}
 return {'mode':mode,'wi':torch.randn(DOMAINS,EXPERTS,HID,DIN,generator=g)*.12,'wo':torch.randn(DOMAINS,EXPERTS,DOUT,HID,generator=g)*.12}
def target(t,x,d,e):
 if t['mode']=='aligned':
  q=torch.bmm(rotate(t['ai'],DIN,True)[d],x.unsqueeze(-1)).squeeze(-1);h=F.gelu(torch.bmm(t['wi'][e],q.unsqueeze(-1)).squeeze(-1));y=torch.bmm(t['wo'][e],h.unsqueeze(-1)).squeeze(-1);return torch.bmm(rotate(t['ao'],DOUT)[d],y.unsqueeze(-1)).squeeze(-1)
 h=F.gelu(torch.bmm(t['wi'][d,e],x.unsqueeze(-1)).squeeze(-1));return torch.bmm(t['wo'][d,e],h.unsqueeze(-1)).squeeze(-1)
def data(t,seed,n):
 g=torch.Generator().manual_seed(seed);x=torch.randn(n,DIN,generator=g);d=torch.randint(DOMAINS,(n,),generator=g);e=torch.randint(EXPERTS,(n,),generator=g);p=torch.randperm(n,generator=g);x=x[p];d=d[p];e=e[p];return x,d,e,target(t,x,d,e)
def payload(m,method):
 b=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},b);cfg=json.dumps({'method':method,'din':DIN,'hidden':HID,'dout':DOUT,'domains':DOMAINS,'experts':EXPERTS},sort_keys=True,separators=(',',':')).encode();return len(b.getvalue())+len(cfg)
def mac(method):return 2*DIN*HID+(int(method[-1])*(DIN+DOUT) if method in ('rank1','rank2') else 0)
def run(mode,w,ts,ins,method,lr):
 t=teacher(mode,ts);x,d,e,y=data(t,ts+1,UPDATES*BATCH);xe,de,ee,ye=data(t,ts+2,8192);m=DomainExperts(method,ins);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(w+150000);idx=torch.randint(len(x),(UPDATES,BATCH),generator=g);st=time.perf_counter()
 for i in range(UPDATES):
  q=idx[i];opt.zero_grad(set_to_none=True);loss=F.mse_loss(m(x[q],d[q],e[q]),y[q]);loss.backward();opt.step()
 wall=time.perf_counter()-st
 with torch.no_grad():
  err=(m(xe,de,ee)-ye).square().mean(1);by={f'{a}:{b}':float(err[(de==a)&(ee==b)].mean()) for a in range(DOMAINS) for b in range(EXPERTS)};st=time.perf_counter()
  for _ in range(20):m(xe[:1024],de[:1024],ee[:1024])
  speed=1024/((time.perf_counter()-st)/20)
 return {'mode':mode,'world':w,'teacher_seed':ts,'initialization_seed':ins,'method':method,'lr':lr,'mse':float(err.mean()),'max_role_mse':max(by.values()),'domain_expert_mse':by,'bytes':payload(m,method),'examples':UPDATES*BATCH,'updates':UPDATES,'active_mac_proxy':mac(method),'train_wall_seconds':wall,'inference_examples_per_second':speed}
def save(path,rows):
 keys=['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'];p=Path(path);new=not p.exists()
 with p.open('a' if not new else 'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n')
  if new:w.writeheader()
  for r in rows:
   note={k:r[k] for k in ('teacher_seed','initialization_seed','lr','domain_expert_mse','inference_examples_per_second')};w.writerow({'condition':r['mode'],'world_or_seed':r['world'],'method':r['method'],'serialized_bytes':r['bytes'],'train_tokens_or_examples':r['examples'],'optimizer_updates':r['updates'],'active_compute_proxy':r['active_mac_proxy'],'wall_time_s':r['train_wall_seconds'],'primary_metric':'heldout_domain_expert_mse','primary_value':r['mse'],'secondary_metric':'max_domain_expert_mse','secondary_value':r['max_role_mse'],'status_note':json.dumps(note,sort_keys=True)})
def development(out):
 rows=[];lrs=(.001,.003,.01)
 for lr in lrs:
  for mode in ('aligned','independent'):
   for j,m in enumerate(METHODS):rows.append(run(mode,150000,1500000+(mode=='independent')*100,15000000+j*1000,m,lr))
 means={lr:[r['mse'] for r in rows if r['lr']==lr] for lr in lrs};sel=min(lrs,key=lambda z:sum(means[z])/len(means[z]));Path(out).mkdir(parents=True,exist_ok=True);save(Path(out)/'RESULTS_CORE.csv',rows);Path(out,'dev_raw.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n');Path(out,'DEV_SELECTION.json').write_text(json.dumps({'selected_learning_rate':sel,'mean_mse_by_lr':{str(k):sum(v)/len(v) for k,v in means.items()},'development_world':150000,'fresh_accessed':False},indent=2,sort_keys=True)+'\n');print(json.dumps({'selected_lr':sel,'means':{str(k):sum(v)/len(v) for k,v in means.items()}},sort_keys=True))
def fresh(out,lr):
 rows=[]
 for w,ts,ins in zip((150001,150002,150003),(1500001,1500002,1500003),(15000001,15000002,15000003)):
  for mode in ('aligned','independent'):
   for j,m in enumerate(METHODS):rows.append(run(mode,w,ts+(mode=='independent')*100,ins+j*1000,m,lr))
 Path(out).mkdir(parents=True,exist_ok=True);save(Path(out)/'RESULTS_CORE.csv',rows);Path(out,'fresh_raw.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n');print(json.dumps({'fresh_rows':len(rows),'mirror_to_independent_mse':{str(w):next(r['mse'] for r in rows if r['world']==w and r['mode']=='aligned' and r['method']=='mirror')/next(r['mse'] for r in rows if r['world']==w and r['mode']=='aligned' and r['method']=='independent') for w in (150001,150002,150003)}}))
def main():
 torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('phase',choices=['development','fresh']);p.add_argument('--out',required=True);p.add_argument('--lr',type=float);a=p.parse_args()
 if a.phase=='development':development(a.out)
 else:
  if a.lr is None:raise SystemExit('--lr required')
  fresh(a.out,a.lr)
if __name__=='__main__':main()
