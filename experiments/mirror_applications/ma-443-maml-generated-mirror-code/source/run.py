#!/usr/bin/env python3
import argparse,hashlib,json,random,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[2];OUT=ROOT/'artifacts';PAY=OUT/'payloads'
DEV=[44300,44301];FRESH=[44310,44311,44312];SEEDS=[0,1,2];OUTER_LRS=[.01,.03];METHODS=['mirror_zero','mirror_encoder','encoder_only','full','leo'];STEPS=[0,1,3,5];D=8;INNER_LR=.05;META_UPDATES=300;TASKS_PER_BATCH=8

def fix(v):random.seed(v);torch.manual_seed(v);torch.set_num_threads(1)
def world(w):
 g=torch.Generator().manual_seed(w);return torch.randn(D,generator=g)*.5
def rot(a):
 q=torch.eye(D);c=a.cos();s=a.sin();q[0,0]=c;q[0,1]=-s;q[1,0]=s;q[1,1]=c;return q
def teacher(base,task):return rot(torch.tensor((task*1.61803398875)%(2*torch.pi)))@base
def samples(w,task,n,offset=0):
 base=world(w);g=torch.Generator().manual_seed(w*100000+task*991+offset+n);x=torch.randn(n,D,generator=g);return x,x@teacher(base,task)
def encode(encoder,x,y):
 # Support-set sufficient statistic; encoder predicts two Mirror/latent coordinates.
 stat=(x.T@y)/x.shape[0];return encoder(stat)
def eff(method,base,code,basis):
 if method=='full':return code
 if method.startswith('mirror'):return rot(code[0])@base+code[1]*torch.nn.functional.one_hot(torch.tensor(2),D).float()
 if method=='leo':return base+basis@code
 return base
def adapt(method,base,encoder,basis,x,y,steps,lr):
 if method=='full':code=base.clone().requires_grad_(True)
 elif method=='mirror_zero':code=torch.zeros(2,requires_grad=True)
 elif method in ('mirror_encoder','encoder_only','leo'):code=encode(encoder,x,y)
 else:code=None
 if method=='encoder_only' or method=='shared':return code
 for _ in range(steps):
  pred=x@eff(method,base,code,basis);loss=(pred-y).square().mean();g=torch.autograd.grad(loss,code,create_graph=False)[0].detach();code=code-lr*g
 return code
def meta_train(w,seed,method,outer_lr):
 fix(w*100+seed*53+sum(map(ord,method)));base=nn.Parameter(torch.randn(D)*.1)
 encoder=nn.Sequential(nn.Linear(D,16),nn.Tanh(),nn.Linear(16,2)) if method in ('mirror_encoder','encoder_only','leo') else nn.Identity()
 basis=nn.Parameter(torch.randn(D,2)*.08) if method=='leo' else torch.zeros(D,2)
 params=[base]+(list(encoder.parameters()) if isinstance(encoder,nn.Module) and not isinstance(encoder,nn.Identity) else [])+([basis] if method=='leo' else [])
 opt=torch.optim.Adam(params,lr=outer_lr);st=time.perf_counter()
 for update in range(META_UPDATES):
  losses=[]
  for j in range(TASKS_PER_BATCH):
   task=1000+(update*TASKS_PER_BATCH+j)%128;x,y=samples(w,task,16,update);qx,qy=samples(w,task,32,update+9001)
   code=adapt(method,base,encoder,basis,x,y,5,INNER_LR);losses.append(((qx@eff(method,base,code,basis)-qy).square().mean()))
  loss=torch.stack(losses).mean();opt.zero_grad();loss.backward();opt.step()
 return base.detach(),{k:v.detach() for k,v in encoder.state_dict().items()} if not isinstance(encoder,nn.Identity) else {},basis.detach(),time.perf_counter()-st
def build_encoder(state):
 e=nn.Sequential(nn.Linear(D,16),nn.Tanh(),nn.Linear(16,2));e.load_state_dict(state);return e
def payload(method,w,task,base,enc_state,basis,code):
 return {'experiment':'MA-443','method':method,'world':w,'task':task,'base':base,'basis':basis if method=='leo' else torch.empty(0),'code':code if code is not None else torch.empty(0),'format':'ma443-task-payload-v1'}
def evaluate(base,enc_state,basis,method,w,task,seed,steps,lr,save_path=None):
 encoder=build_encoder(enc_state) if enc_state else nn.Identity();sx,sy=samples(w,task,8,seed+311);qx,qy=samples(w,task,128,seed+771);t0=time.perf_counter();code=adapt(method,base,encoder,basis,sx,sy,steps,lr)
 with torch.no_grad():err=((qx@eff(method,base,code,basis)-qy).square().mean().sqrt()/(qy.square().mean().sqrt()+1e-12)).item()
 wall=time.perf_counter()-t0
 if save_path:
  torch.save(payload(method,w,task,base,enc_state,basis,code),save_path);b=save_path.read_bytes();return err,wall,len(b),hashlib.sha256(b).hexdigest(),str(save_path.relative_to(REPO))
 temp=payload(method,w,task,base,enc_state,basis,code);buf=__import__('io').BytesIO();torch.save(temp,buf);return err,wall,len(buf.getvalue()),'', ''
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','fresh'],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);PAY.mkdir(parents=True,exist_ok=True);worlds=DEV if a.phase=='development' else FRESH;rows=[];selected=json.loads((OUT/'development_selection.json').read_text())['outer_lr_by_method'] if a.phase=='fresh' else {}
 for w in worlds:
  for seed in SEEDS:
   for method in METHODS:
    if a.phase=='development':
     scores={lr:[] for lr in OUTER_LRS}
     for outer_lr in OUTER_LRS:
      base,enc,basis,meta_wall=meta_train(w,seed,method,outer_lr)
      for task in range(12):
       for steps in STEPS:
        err,wall,nbytes,_,_=evaluate(base,enc,basis,method,w,task,seed+100,steps,INNER_LR)
        scores[outer_lr].append(err)
        rows.append({'condition':'development','world_or_seed':f'{w}-{seed}-{task}-{steps}','method':method,'serialized_bytes':nbytes,'train_tokens_or_examples':META_UPDATES*TASKS_PER_BATCH*16,'optimizer_updates':META_UPDATES,'active_compute_proxy':f'{steps*D*2} MAC/inner-adapt','wall_time_s':round(wall,6),'primary_metric':'query_NRMSE','primary_value':err,'secondary_metric':'refinement_steps','secondary_value':steps,'status_note':f'outer_lr={outer_lr}; inner_lr={INNER_LR}; meta_wall={meta_wall:.4f}'})
     selected[method]=min(OUTER_LRS,key=lambda lr:sum(scores[lr])/len(scores[lr]))
    else:
     base,enc,basis,meta_wall=meta_train(w,seed,method,selected[method])
     encoder_path=PAY/f'meta_{w}_{seed}_{method}_encoder.pt' if enc else None
     if encoder_path:
      torch.save({'method':method,'world':w,'state':enc},encoder_path);eb=encoder_path.stat().st_size;eh=hashlib.sha256(encoder_path.read_bytes()).hexdigest()
     else: eb=0;eh='';encoder_path=None
     for task_offset in range(20):
      task=2000+task_offset
      for steps in STEPS:
       path=PAY/f'fresh_{w}_{seed}_{task}_{method}_{steps}.pt'
       err,wall,nbytes,h,p=evaluate(base,enc,basis,method,w,task,seed+100,steps,INNER_LR,path)
       amortized={n:round(nbytes+eb/max(1,n)) for n in [1,20,100]}
       note=f'outer_lr={selected[method]}; inner_lr={INNER_LR}; amortized_bytes={amortized}; encoder_bytes={eb}; meta_wall={meta_wall:.4f}; hash={h}; payload={p}'
       if encoder_path: note+=f'; encoder_hash={eh}; encoder_path={encoder_path.relative_to(REPO)}'
       rows.append({'condition':'fresh','world_or_seed':f'{w}-{seed}-{task}-{steps}','method':method,'serialized_bytes':nbytes,'train_tokens_or_examples':META_UPDATES*TASKS_PER_BATCH*16,'optimizer_updates':META_UPDATES,'active_compute_proxy':f'{steps*D*2} MAC/inner-adapt + encoder','wall_time_s':round(wall,6),'primary_metric':'query_NRMSE','primary_value':err,'secondary_metric':'refinement_steps','secondary_value':steps,'status_note':note})
 if a.phase=='development':
  selected['inner_lr']=INNER_LR;(OUT/'development_selection.json').write_text(json.dumps({'outer_lr_by_method':selected,'inner_lr':INNER_LR,'rule':'lowest mean query NRMSE across development tasks and 0/1/3/5 updates','fresh_worlds_not_accessed':True},indent=2)+'\n')
 (OUT/f'{a.phase}_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'phase':a.phase,'selected':selected,'rows':len(rows)},indent=2))
if __name__=='__main__':main()
