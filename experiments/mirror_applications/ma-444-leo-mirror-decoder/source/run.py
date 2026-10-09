#!/usr/bin/env python3
import argparse,hashlib,json,random,time,io
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[2];OUT=ROOT/'artifacts';PAY=OUT/'payloads'
DEV=[44400,44401];FRESH=[44410,44411,44412];SEEDS=[0,1,2];LRS=[.01,.03];METHODS=['shared','mirror','leo','full'];STEPS=[0,1,3,5];D=8;INNER_LR=.05;UPDATES=300;TASK_BATCH=8

def fix(s):random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def world(w):
 g=torch.Generator().manual_seed(w);return torch.randn(D,generator=g)*.5
def rot(a):
 q=torch.eye(D);c=a.cos();s=a.sin();q[0,0]=c;q[0,1]=-s;q[1,0]=s;q[1,1]=c;return q
def angle(task):return torch.tensor((task*1.61803398875)%(2*torch.pi))
def target(base,task):return rot(angle(task))@base
def samples(w,task,n,off=0):
 g=torch.Generator().manual_seed(w*100000+task*991+off+n);x=torch.randn(n,D,generator=g);return x,x@target(world(w),task)
def effective(method,base,code,basis):
 if method=='full':return code
 if method=='mirror':return rot(code[0])@base+code[1]*torch.nn.functional.one_hot(torch.tensor(2),D).float()
 if method=='leo':return base+basis@code
 return base
def adapt(method,base,basis,x,y,steps,lr):
 if method=='shared':return None
 code=base.clone().requires_grad_(True) if method=='full' else torch.zeros(2,requires_grad=True)
 for _ in range(steps):
  loss=(x@effective(method,base,code,basis)-y).square().mean();g=torch.autograd.grad(loss,code)[0].detach();code=code-lr*g
 return code
def train_meta(w,seed,method,outer_lr):
 fix(w*100+seed*79+sum(map(ord,method)));base=nn.Parameter(torch.randn(D)*.1);basis=nn.Parameter(torch.randn(D,2)*.08) if method=='leo' else torch.zeros(D,2);params=[base]+([basis] if method=='leo' else []);opt=torch.optim.Adam(params,lr=outer_lr);start=time.perf_counter()
 for it in range(UPDATES):
  lossvec=[]
  for j in range(TASK_BATCH):
   task=1000+(it*TASK_BATCH+j)%128;x,y=samples(w,task,16,it);qx,qy=samples(w,task,32,it+9001);code=adapt(method,base,basis,x,y,5,INNER_LR);lossvec.append((qx@effective(method,base,code,basis)-qy).square().mean())
  loss=torch.stack(lossvec).mean();opt.zero_grad();loss.backward();opt.step()
 return base.detach(),basis.detach(),time.perf_counter()-start
def evaluate(base,basis,method,w,task,seed,steps,lr,path=None,decoder_bytes=0):
 sx,sy=samples(w,task,8,seed+311);qx,qy=samples(w,task,128,seed+771);start=time.perf_counter();code=adapt(method,base,basis,sx,sy,steps,lr)
 with torch.no_grad():err=((qx@effective(method,base,code,basis)-qy).square().mean().sqrt()/(qy.square().mean().sqrt()+1e-12)).item()
 wall=time.perf_counter()-start;obj={'method':method,'world':w,'task':task,'base':base,'code':code if code is not None else torch.empty(0),'format':'ma444-task-v1'}
 if path:
  torch.save(obj,path);buf=path.read_bytes();size=len(buf);sha=hashlib.sha256(buf).hexdigest();rel=str(path.relative_to(REPO))
 else:
  b=io.BytesIO();torch.save(obj,b);size=len(b.getvalue());sha='';rel=''
 return err,wall,size,sha,rel
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','fresh'],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);PAY.mkdir(parents=True,exist_ok=True);worlds=DEV if a.phase=='development' else FRESH;rows=[];chosen=json.loads((OUT/'development_selection.json').read_text())['outer_lr'] if a.phase=='fresh' else {}
 for w in worlds:
  for seed in SEEDS:
   for method in METHODS:
    if a.phase=='development':
     bylr={lr:[] for lr in LRS}
     for lr in LRS:
      base,basis,metawall=train_meta(w,seed,method,lr)
      for task in range(12):
       for steps in STEPS:
        err,wall,size,_,_=evaluate(base,basis,method,w,task,seed+100,steps,INNER_LR);bylr[lr].append(err);rows.append({'condition':'development','world_or_seed':f'{w}-{seed}-{task}-{steps}','method':method,'serialized_bytes':size,'train_tokens_or_examples':UPDATES*TASK_BATCH*16,'optimizer_updates':UPDATES,'active_compute_proxy':f'{steps*D*2} MAC/adapt','wall_time_s':round(wall,6),'primary_metric':'query_NRMSE','primary_value':err,'secondary_metric':'refinement_steps','secondary_value':steps,'status_note':f'outer_lr={lr}; meta_wall={metawall:.3f}'})
     chosen[method]=min(LRS,key=lambda lr:sum(bylr[lr])/len(bylr[lr]))
    else:
     base,basis,metawall=train_meta(w,seed,method,chosen[method]);decpath=PAY/f'meta_{w}_{seed}_leo_decoder.pt' if method=='leo' else None
     if decpath:torch.save({'world':w,'state':basis},decpath);db=decpath.stat().st_size;dh=hashlib.sha256(decpath.read_bytes()).hexdigest();drel=str(decpath.relative_to(REPO))
     else:db=0;dh='';drel=''
     for toff in range(20):
      task=2000+toff
      for steps in STEPS:
       path=PAY/f'fresh_{w}_{seed}_{task}_{method}_{steps}.pt';err,wall,size,h,rel=evaluate(base,basis,method,w,task,seed+100,steps,INNER_LR,path,db);am={n:round(size+db/max(1,n)) for n in [1,20,100]};note=f'outer_lr={chosen[method]}; amortized_bytes={am}; decoder_bytes={db}; hash={h}; payload={rel}; meta_wall={metawall:.3f}'
       if decpath:note+=f'; decoder_hash={dh}; decoder_path={drel}'
       rows.append({'condition':'fresh','world_or_seed':f'{w}-{seed}-{task}-{steps}','method':method,'serialized_bytes':size,'train_tokens_or_examples':UPDATES*TASK_BATCH*16,'optimizer_updates':UPDATES,'active_compute_proxy':f'{steps*D*2} MAC/adapt','wall_time_s':round(wall,6),'primary_metric':'query_NRMSE','primary_value':err,'secondary_metric':'refinement_steps','secondary_value':steps,'status_note':note})
 if a.phase=='development':chosen['inner_lr']=INNER_LR;(OUT/'development_selection.json').write_text(json.dumps({'outer_lr':chosen,'inner_lr':INNER_LR,'rule':'lowest dev mean NRMSE across tasks and steps','fresh_not_accessed':True},indent=2)+'\n')
 (OUT/f'{a.phase}_runs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));print(json.dumps({'phase':a.phase,'selected':chosen,'rows':len(rows)},indent=2))
if __name__=='__main__':main()
