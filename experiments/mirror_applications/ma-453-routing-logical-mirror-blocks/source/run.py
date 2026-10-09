#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,math,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';DEV=[45300,45301];FRESH=[45310,45311,45312];SEEDS=[0,1,2];KS=[4,8]
def task(w,s,tid):
 g=torch.Generator().manual_seed(w*1000003+s*997+tid*53);a=.5+1.5*torch.rand((),generator=g);b=-.5+torch.rand((),generator=g);x=-2+4*torch.rand(32,generator=g);qx=-2+4*torch.rand(256,generator=torch.Generator().manual_seed(w*99131+s*71+tid*17));return float(a),float(b),x,torch.tanh(a*x+b),qx,torch.tanh(a*qx+b)
def protofit(points,k):
 centers=points[torch.linspace(0,len(points)-1,k).long()].clone()
 for _ in range(40):
  dist=torch.cdist(points,centers);ids=dist.argmin(1)
  for j in range(k):
   if (ids==j).any():centers[j]=points[ids==j].mean(0)
 return centers
def fitcode(x,y,steps=80):
 p=torch.nn.Parameter(torch.tensor([1.,0.]));o=torch.optim.Adam([p],lr=.04)
 for _ in range(steps):o.zero_grad();torch.nn.functional.mse_loss(torch.tanh(p[0]*x+p[1]),y).backward();o.step()
 return p.detach()
def nrmse(pred,y):return float(((pred-y).square().mean().sqrt())/(y.square().mean().sqrt()+1e-12))
def evaluate(w,s,tid,centers):
 a,b,x,y,qx,qy=task(w,s,tid);out={}
 t=time.perf_counter();code=fitcode(x,y);out['mirror']={'nrmse':nrmse(torch.tanh(code[0]*qx+code[1]),qy),'code':code,'wall':time.perf_counter()-t}
 t=time.perf_counter();ind=fitcode(x,y);out['independent']={'nrmse':nrmse(torch.tanh(ind[0]*qx+ind[1]),qy),'code':ind,'wall':time.perf_counter()-t}
 for k,c in centers.items():
  t=time.perf_counter();losses=[float(torch.nn.functional.mse_loss(torch.tanh(aa*x+bb),y)) for aa,bb in c];idx=min(range(k),key=lambda i:losses[i]);aa,bb=c[idx];out[f'router{k}']={'nrmse':nrmse(torch.tanh(aa*qx+bb),qy),'index':idx,'wall':time.perf_counter()-t}
 out['shared']={'nrmse':nrmse(torch.tanh(qx),qy),'wall':0.};return out

def pack(method,items,centers,N):
 sub=items[:N];obj={'format':'ma453-v1','method':method,'N':N}
 if method=='mirror':obj.update(shared_block='tanh',codes=torch.stack([x['out']['mirror']['code'] for x in sub]))
 elif method=='independent':obj.update(private_blocks=torch.stack([x['out']['independent']['code'] for x in sub]))
 elif method=='shared':obj.update(shared_block='tanh')
 elif method.startswith('router'):
  k=int(method.replace('router',''));obj.update(prototypes=centers[k],route_indices=torch.tensor([x['out'][method]['index'] for x in sub],dtype=torch.uint8))
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','fresh'],required=True);args=ap.parse_args();ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True)
 if args.phase=='development':
  pts=torch.tensor([[task(w,s,t)[0],task(w,s,t)[1]] for w in DEV for s in SEEDS for t in range(100,164)])
  centers={k:protofit(pts,k) for k in KS};scores={k:[] for k in KS}
  for w in DEV:
   for s in SEEDS:
    for tid in range(100,164):
     _,_,x,y,qx,qy=task(w,s,tid)
     for k,c in centers.items():
      ix=min(range(k),key=lambda i:float(torch.nn.functional.mse_loss(torch.tanh(c[i,0]*x+c[i,1]),y)));scores[k].append(nrmse(torch.tanh(c[ix,0]*qx+c[ix,1]),qy))
  chosen=min(KS,key=lambda k:statistics.mean(scores[k]));torch.save(centers,PAY/'router_prototypes.pt');(ART/'development_selection.json').write_text(json.dumps({'router_K':chosen,'scores':{str(k):statistics.mean(v) for k,v in scores.items()},'fresh_untouched':True},indent=2)+'\n');print(json.dumps({'selected_K':chosen,'scores':{k:statistics.mean(v) for k,v in scores.items()}}));return
 sel=json.loads((ART/'development_selection.json').read_text());centers=torch.load(PAY/'router_prototypes.pt',weights_only=True);K=sel['router_K'];rows=[]
 for w in FRESH:
  for s in SEEDS:
   items=[]
   for tid in range(3000,3064):items.append({'world':w,'seed':s,'tid':tid,'out':evaluate(w,s,tid,{K:centers[K]})})
   for method in ['shared',f'router{K}','mirror','independent']:
    for N in [1,20,64]:
     data=pack(method,items,centers,N);path=PAY/f'{w}_{s}_{method}_N{N}.pt';path.write_bytes(data)
     for group in [0]:
      rows.append({'world':w,'seed':s,'method':method,'n':N,'tasks':N,'nrmse_mean':statistics.mean(x['out'][method]['nrmse'] for x in items[:N]),'payload_bytes':len(data),'bytes_per_task':len(data)/N,'compute_MAC_per_task':(K*32 if method.startswith('router') else 80*32*2 if method in ('mirror','independent') else 0),'query_wall_seconds_mean':statistics.mean(x['out'][method]['wall'] for x in items[:N]),'hash':hashlib.sha256(data).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with open(ART/'fresh_runs.jsonl','w') as f:f.write(''.join(json.dumps(r)+'\n' for r in rows))
 with open(ROOT/'RESULTS_CORE.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':'fresh','rows':len(rows),'router_K':K}))
if __name__=='__main__':main()
