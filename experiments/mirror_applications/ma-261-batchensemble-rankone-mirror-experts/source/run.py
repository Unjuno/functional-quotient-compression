import csv,hashlib,io,json,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];E=4;D=2;DEV=[26100,26101];FRESH=[26110,26111,26112];SEEDS=[0,1,2];STEPS=500;torch.set_num_threads(2)
def teacher(w,s):
 g=torch.Generator().manual_seed(w*100003+s*7919+261);base=torch.randn(2,2,generator=g)*.5;u=F.normalize(torch.randn(2,generator=g),dim=0);v=F.normalize(torch.randn(2,generator=g),dim=0);alpha=torch.tensor([-1.2,-.4,.4,1.2]);alpha=alpha[torch.randperm(E,generator=g)];return base,u,v,alpha
def data(w,s,base,u,v,a):
 g=torch.Generator().manual_seed(w*100003+s*7919+88);centers=torch.tensor([[-2.,-2.],[-2.,2.],[2.,-2.],[2.,2.]])+torch.randn(4,2,generator=g)*.15
 def make(n):
  y=torch.randint(E,(n,),generator=g);x=centers[y]+torch.randn(n,2,generator=g)*.65;W=base[None,:,:]+a[:,None,None]*torch.outer(u,v);target=torch.einsum('nd,edh->neh',x,W)[torch.arange(n),y];return x,y,target
 return make(2048),make(1024)
def router_fit(x,y,seed):
 torch.manual_seed(seed);g=nn.Linear(D,E);opt=torch.optim.Adam(g.parameters(),lr=.03)
 for _ in range(300):opt.zero_grad();loss=F.cross_entropy(g(x),y);loss.backward();opt.step()
 return g
def weights_from(method,p):
 if method=='standard_moe':return p['W']
 if method=='tied':return p['W'].expand(E,-1,-1)
 if method=='batchensemble':return torch.stack([torch.diag(p['s'][e])@p['W']@torch.diag(p['r'][e]) for e in range(E)])
 if method=='mirror':return p['base'][None,:,:]+1.5*torch.tanh(p['m'])[:,None,None]*torch.outer(p['u'],p['v'])[None,:,:]
 return p['base'][None,:,:]+p['alpha'][:,None,None]*torch.outer(p['u'],p['v'])[None,:,:]
def init(method,seed):
 torch.manual_seed(seed)
 if method=='standard_moe':return {'W':nn.Parameter(torch.randn(E,2,2)*.1)}
 if method=='tied':return {'W':nn.Parameter(torch.randn(2,2)*.1)}
 if method=='batchensemble':return {'W':nn.Parameter(torch.randn(2,2)*.1),'r':nn.Parameter(torch.ones(E,2)),'s':nn.Parameter(torch.ones(E,2))}
 if method=='mirror':return {'base':nn.Parameter(torch.randn(2,2)*.1),'u':nn.Parameter(F.normalize(torch.randn(2),dim=0)),'v':nn.Parameter(F.normalize(torch.randn(2),dim=0)),'m':nn.Parameter(torch.zeros(E))}
 return {'base':nn.Parameter(torch.randn(2,2)*.1),'u':nn.Parameter(F.normalize(torch.randn(2),dim=0)),'v':nn.Parameter(F.normalize(torch.randn(2),dim=0)),'alpha':nn.Parameter(torch.zeros(E))}
def fit(method,x,y,target,seed):
 p=init(method,seed);opt=torch.optim.Adam(list(p.values()),lr=.02);t=time.perf_counter()
 for _ in range(STEPS):
  opt.zero_grad();W=weights_from(method,p);pred=torch.einsum('nd,edh->neh',x,W)[torch.arange(len(y)),y];loss=(pred-target).square().mean();loss.backward();opt.step()
 return {k:v.detach() for k,v in p.items()},time.perf_counter()-t
def serialize(o):b=io.BytesIO();torch.save(o,b);return b.getvalue()
def flatten(state):
 keys=sorted(state);vals=[state[k].detach().contiguous().reshape(-1) for k in keys];shapes=[list(state[k].shape) for k in keys];return torch.cat(vals),keys,shapes
def unflatten(flat,keys,shapes):
 out={};off=0
 for k,shape in zip(keys,shapes):
  n=1
  for d in shape:n*=d
  out[k]=flat[off:off+n].reshape(shape);off+=n
 return out
def run(phase):
 rows=[];worlds=DEV if phase=='development' else FRESH
 for w in worlds:
  for seed in SEEDS:
   base,u,v,a=teacher(w,seed);(xt,yt,tt),(xv,yv,tv)=data(w,seed,base,u,v,a);router=router_fit(xt,yt,seed+w);routed=router(xv).argmax(-1);racc=float((routed==yv).float().mean())
   for method in ['standard_moe','tied','batchensemble','mirror','generic_rank1']:
    p,sec=fit(method,xt,yt,tt,seed+w+13);W=weights_from(method,p);allpred=torch.einsum('nd,edh->neh',xv,W);oracle=allpred[torch.arange(len(yv)),yv];pred=allpred[torch.arange(len(yv)),routed];metric=lambda z:float((z-tv).square().mean().sqrt()/tv.square().mean().sqrt());ef,ek,es=flatten(p);rf,rk,rs=flatten(router.state_dict());blob=serialize({'method':method,'expert_flat':ef,'expert_keys':ek,'expert_shapes':es,'router_flat':rf,'router_keys':rk,'router_shapes':rs,'metadata':{'E':E,'active_k':1,'steps':STEPS,'dtype':'float32'}});path=ROOT/'artifacts'/'payloads'/f'{w}_{seed}_{method}.pt';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob);td=time.perf_counter();loaded=torch.load(io.BytesIO(blob),weights_only=True);decoded=unflatten(loaded['expert_flat'],loaded['expert_keys'],loaded['expert_shapes']);_w=weights_from(method,decoded);decode=time.perf_counter()-td
    rows.append({'world':w,'seed':seed,'method':method,'routed_nrmse':metric(pred),'oracle_route_nrmse':metric(oracle),'router_accuracy':racc,'active_experts':int(routed.unique().numel()),'payload_bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2])),'train_seconds':sec,'decode_seconds':decode,'updates':STEPS,'active_k':1})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
