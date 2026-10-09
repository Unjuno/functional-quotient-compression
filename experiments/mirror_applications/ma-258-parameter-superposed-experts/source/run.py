import csv,hashlib,io,json,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];D=32;E=8;SEEDS=[0,1,2];DEV=[25800,25801];FRESH=[25810,25811,25812];torch.set_num_threads(2)
def make_tasks(w,s,stratum):
 g=torch.Generator().manual_seed(w*100003+s*7919+(17 if stratum=='aligned' else 29));base=torch.randn(D,D,generator=g)/D**.5;Ws=[]
 for e in range(E):
  if stratum=='aligned':
   a=(torch.rand((),generator=g)*2-1)*.7;c=torch.cos(a);q=torch.sin(a);R=torch.eye(D);R[0,0]=c;R[0,1]=-q;R[1,0]=q;R[1,1]=c;W=R@base
  else:
   U=torch.randn(D,2,generator=g)/D**.5;V=torch.randn(D,2,generator=g)/D**.5;W=base+U@V.T
  Ws.append(W)
 return base,torch.stack(Ws)
def data(w,s,stratum,W):
 g=torch.Generator().manual_seed(w*100003+s*7919+37+(1 if stratum=='aligned' else 3));x=torch.randn(E,768,D,generator=g);y=torch.einsum('end,edh->enh',x,W);return x[:,:256],y[:,:256],x[:,256:],y[:,256:]
def rotate(base,a):
 R=torch.eye(D);c=torch.cos(a);s=torch.sin(a);R[0,0]=c;R[0,1]=-s;R[1,0]=s;R[1,1]=c;return R@base
def fit_angle(base,x,y):
 z=torch.nn.Parameter(torch.zeros(()));opt=torch.optim.Adam([z],lr=.03)
 for _ in range(500):
  opt.zero_grad();a=.7*torch.tanh(z);loss=(x@rotate(base,a)-y).square().mean();loss.backward();opt.step()
 return (.7*torch.tanh(z)).detach()
def rank2(base,W):
 d=W-base;u,s,v=torch.linalg.svd(d,full_matrices=False);return u[:,:2]*s[:2],v[:2].T
def nrmse(pred,target):return float((pred-target).square().mean().sqrt()/target.square().mean().sqrt())
def serialize(o):b=io.BytesIO();torch.save(o,b);return b.getvalue()
def run(phase):
 rows=[];worlds=DEV if phase=='development' else FRESH
 for w in worlds:
  for seed in SEEDS:
   for stratum in ['aligned','random_rank2']:
    base,W=make_tasks(w,seed,'aligned' if stratum=='aligned' else 'random');xt,yt,xv,yv=data(w,seed,'aligned' if stratum=='aligned' else 'random',W)
    t=time.perf_counter();ang=torch.stack([fit_angle(base,xt[e],yt[e]) for e in range(E)]);mt=time.perf_counter()-t;M=torch.stack([rotate(base,a) for a in ang])
    t=time.perf_counter();fac=[rank2(base,W[e]) for e in range(E)];rt=time.perf_counter()-t;LR=torch.stack([base+u@v.T for u,v in fac])
    C=torch.randint(0,2,(E,D,D),generator=torch.Generator().manual_seed(w*100+seed+55)).float()*2-1;S=(C*W).sum(0);PSP=C*S;tied=W.mean(0).expand(E,-1,-1)
    objs={'untied':(W,{'experts':W,'metadata':{'E':E}}),'hard_tied':(tied,{'expert':W.mean(0),'metadata':{'E':E}}),'native_psp':(PSP,{'superposed_tensor':S,'contexts':C,'metadata':{'E':E,'unbinding':'hadamard'}}),'mirror_givens':(M,{'shared_expert':base,'angles':ang,'metadata':{'plane':[0,1],'E':E}}),'rank2_private':(LR,{'shared_expert':base,'factors':fac,'metadata':{'rank':2,'E':E}})}
    for name,(pred,obj) in objs.items():
     blob=serialize(obj);p=ROOT/'artifacts'/'payloads'/f'{w}_{seed}_{stratum}_{name}.pt';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(blob);errs=[nrmse(xv[e]@pred[e],yv[e]) for e in range(E)]
     rows.append({'world':w,'seed':seed,'stratum':stratum,'method':name,'mean_expert_nrmse':sum(errs)/E,'max_expert_nrmse':max(errs),'payload_bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'path':str(p.relative_to(ROOT.parents[2])),'fit_seconds':mt if name=='mirror_givens' else rt if name=='rank2_private' else 0.,'experts':E})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
