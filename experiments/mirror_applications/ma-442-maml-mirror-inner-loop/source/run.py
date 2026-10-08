#!/usr/bin/env python3
import argparse,hashlib,json,random,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[2];OUT=ROOT/'artifacts';PAY=OUT/'payloads'
DEV=[44200,44201];FRESH=[44210,44211,44212];SEEDS=[0,1,2];LRS=[.01,.03];METHODS=['shared','mirror','full','lora'];D=8;UPDATES=300;TASKS=8;INNER=5;ILR=.05

def fix(s):random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def world(w):
 g=torch.Generator().manual_seed(w);return torch.randn(D,generator=g)*.5
def rot(a):
 q=torch.eye(D);c=a.cos();s=a.sin();q[0,0]=c;q[0,1]=-s;q[1,0]=s;q[1,1]=c;return q
def teacher(base,ang):return rot(ang)@base
def samples(w,task,n,offset=0):
 base=world(w);ang=((task*1.61803398875+offset*.13)%(2*torch.pi));wg=torch.Generator().manual_seed(w*100000+task*991+offset+n);x=torch.randn(n,D,generator=wg);return x,x@teacher(base,torch.tensor(ang))
def eff(method,base,code,basis):
 if method=='full':return code
 if method=='mirror':return rot(code[0])@base+code[1]*torch.nn.functional.one_hot(torch.tensor(2),D).float()
 if method=='lora':return base+basis@code
 return base
def adapt(method,base,basis,x,y,steps,lr):
 if method=='shared':return None
 code=base.clone().requires_grad_(True) if method=='full' else torch.zeros(2,requires_grad=True)
 for _ in range(steps):
  pred=x@eff(method,base,code,basis);loss=(pred-y).square().mean();g=torch.autograd.grad(loss,code,create_graph=False)[0].detach();code=code-lr*g
 return code
def query_loss(method,base,basis,code,x,y):return (x@eff(method,base,code,basis)-y).square().mean()
def meta_train(w,s,method,outer_lr,phase):
 fix(w*100+s*71+sum(map(ord,method)));base=nn.Parameter(torch.randn(D)*.1);basis=nn.Parameter(torch.randn(D,2)*.08) if method=='lora' else torch.zeros(D,2);opt=torch.optim.Adam([base]+([basis] if method=='lora' else []),lr=outer_lr);st=time.perf_counter()
 for step in range(UPDATES):
  losses=[]
  for j in range(TASKS):
   task=(step*TASKS+j)%128;x,y=samples(w,task,16,offset=step);qx,qy=samples(w,task,32,offset=step+9001);code=adapt(method,base,basis,x,y,INNER,ILR);losses.append(query_loss(method,base,basis,code,qx,qy))
  loss=torch.stack(losses).mean();opt.zero_grad();loss.backward();opt.step()
 return base.detach(),basis.detach(),time.perf_counter()-st
def evaluate(base,basis,method,w,task,seed,lr):
 sx,sy=samples(w,task,8,seed+311);qx,qy=samples(w,task,128,seed+771);st=time.perf_counter();code=adapt(method,base,basis,sx,sy,INNER,lr)
 with torch.no_grad():err=((qx@eff(method,base,code,basis)-qy).square().mean().sqrt()/(qy.square().mean().sqrt()+1e-12)).item()
 wall=time.perf_counter()-st
 payload={'method':method,'world':w,'task':task,'base':base,'basis':basis if method=='lora' else torch.empty(0),'adapted':code,'format':'ma442-state-v1'};p=PAY/f'fresh_{w}_{seed}_{task}_{method}.pt';torch.save(payload,p);b=p.read_bytes();return err,wall,len(b),hashlib.sha256(b).hexdigest(),str(p.relative_to(REPO))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','fresh'],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);PAY.mkdir(parents=True,exist_ok=True);worlds=DEV if a.phase=='development' else FRESH;rows=[];sel=json.loads((OUT/'development_selection.json').read_text())['selected_outer_lr'] if a.phase=='fresh' else {}
 for w in worlds:
  for s in SEEDS:
   for method in METHODS:
    if a.phase=='development':
     scores={lr:[] for lr in LRS}
     for lr in LRS:
      base,basis,outerwall=meta_train(w,s,method,lr,a.phase)
      for t in range(12):
       err,wall,nbytes,h,p=evaluate(base,basis,method,w,t,100+s,ILR);scores[lr].append(err);rows.append({'condition':'development','world_or_seed':f'{w}-{s}-{t}','method':method,'serialized_bytes':nbytes,'train_tokens_or_examples':UPDATES*TASKS*16,'optimizer_updates':UPDATES,'active_compute_proxy':f'{INNER*D*2} MAC/adaptation-step','wall_time_s':round(wall,6),'primary_metric':'query_NRMSE','primary_value':err,'secondary_metric':'inner_updates','secondary_value':INNER,'status_note':f'outer_lr={lr}; inner_lr={ILR}; outer_wall={outerwall:.4f}; hash={h}; payload={p}'})
     sel[method]=min(LRS,key=lambda z:sum(scores[z])/len(scores[z]))
    else:
     base,basis,outerwall=meta_train(w,s,method,sel[method],a.phase)
     for t in range(20):
      err,wall,nbytes,h,p=evaluate(base,basis,method,w,t,100+s,ILR);rows.append({'condition':'fresh','world_or_seed':f'{w}-{s}-{t}','method':method,'serialized_bytes':nbytes,'train_tokens_or_examples':UPDATES*TASKS*16,'optimizer_updates':UPDATES,'active_compute_proxy':f'{INNER*D*2} MAC/adaptation-step','wall_time_s':round(wall,6),'primary_metric':'query_NRMSE','primary_value':err,'secondary_metric':'inner_updates','secondary_value':INNER,'status_note':f'outer_lr={sel[method]}; inner_lr={ILR}; outer_wall={outerwall:.4f}; hash={h}; payload={p}'})
 if a.phase=='development':(OUT/'development_selection.json').write_text(json.dumps({'selected_outer_lr':sel,'inner_lr':ILR,'rule':'lowest dev task mean query NRMSE','fresh_worlds_not_accessed':True},indent=2)+'\n')
 (OUT/f'{a.phase}_runs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));print(json.dumps({'phase':a.phase,'selected_outer_lr':sel,'rows':len(rows)},indent=2))
if __name__=='__main__':main()
