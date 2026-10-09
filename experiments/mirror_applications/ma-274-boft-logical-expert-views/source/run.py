"""MA-274: routed synthetic expert bank with BOFT-view and simple controls."""
import argparse,csv,hashlib,json,math,struct,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];D,E=16,4;DEV,FRESH=[27400,27401],[27410,27411,27412];SEEDS=[0,1,2];STEPS=400
torch.set_num_threads(2)

def make(world,seed):
 g=torch.Generator().manual_seed(world*100003+seed*7919+274);w=torch.randn(D,D,generator=g)/math.sqrt(D)
 centers=torch.tensor([[-2.,-2.],[-2.,2.],[2.,-2.],[2.,2.]])+0.1*torch.randn(4,2,generator=g)
 def data(n,off):
  q=torch.Generator().manual_seed(world*100003+seed*7919+off);y=torch.randint(E,(n,),generator=q);z=centers[y]+.65*torch.randn(n,2,generator=q);x=torch.cat([z,torch.randn(n,D-2,generator=q)],1);h=torch.tanh(x@w);return x,h,y
 xt,ht,yt=data(2048,11);xv,hv,yv=data(1024,17)
 # Expert teachers share a base and vary by a 2D Givens transform on feature channels.
 base=torch.randn(D,D,generator=g)*.35;angles=torch.tensor([-.9,-.3,.3,.9]);teachers=[]
 for a in angles:
  q=torch.eye(D);c,s=torch.cos(a),torch.sin(a);q[0,0]=c;q[1,1]=c;q[0,1]=-s;q[1,0]=s;teachers.append(base@q)
 return (xt,ht,yt),(xv,hv,yv),torch.stack(teachers)

def router_fit(x,y,seed):
 torch.manual_seed(seed);m=nn.Linear(2,E);opt=torch.optim.Adam(m.parameters(),lr=.03)
 for _ in range(300):opt.zero_grad();loss=F.cross_entropy(m(x[:,:2]),y);loss.backward();opt.step()
 return m

def rotate(qtheta):
 c,s=torch.cos(qtheta),torch.sin(qtheta);q=torch.eye(D);q[0,0]=c;q[1,1]=c;q[0,1]=-s;q[1,0]=s;return q

def boft_view(code):
 # Four disjoint two-channel Givens stages, a small true butterfly-style factor bank.
 q=torch.eye(D)
 for stage in range(4):
  b=torch.eye(D);offset=stage
  for j in range(0,D,2):
   i=(j+offset)%D;k=(i+1)%D;theta=code[stage,j//2];c,s=torch.cos(theta),torch.sin(theta)
   b[i,i]=c;b[k,k]=c;b[i,k]=-s;b[k,i]=s
  q=q@b
 return q

def weights(method,p):
 if method=='standard_moe':return p['W']
 if method=='tied':return p['W'].expand(E,-1,-1)
 if method=='ia3_gate':return p['W'][None,:,:]*p['g'][:,None,:]
 if method=='rank1':return p['base'][None,:,:]+p['u'][:,None,None]*p['v'][None,None,:]
 if method=='givens_mirror':return torch.stack([p['base']@rotate(a) for a in p['angle']])
 if method=='boft_mirror':return torch.stack([p['base']@boft_view(a) for a in p['angle']])
 if method=='givens_mirror':return torch.stack([p['base']@rotate(a) for a in p['angle']])
 return p['W']

def init(method,seed):
 torch.manual_seed(seed)
 if method=='standard_moe':return {'W':nn.Parameter(torch.randn(E,D,D)*.05)}
 if method=='tied':return {'W':nn.Parameter(torch.randn(D,D)*.05)}
 if method=='ia3_gate':return {'W':nn.Parameter(torch.randn(D,D)*.05),'g':nn.Parameter(torch.ones(E,D))}
 if method=='rank1':return {'base':nn.Parameter(torch.randn(D,D)*.05),'u':nn.Parameter(torch.zeros(E)),'v':nn.Parameter(torch.randn(D)*.05)}
 if method=='boft_mirror':return {'base':nn.Parameter(torch.randn(D,D)*.05),'angle':nn.Parameter(torch.zeros(E,4,D//2))}
 if method=='givens_mirror':return {'base':nn.Parameter(torch.randn(D,D)*.05),'angle':nn.Parameter(torch.zeros(E))}
 return {'base':nn.Parameter(torch.randn(D,D)*.05),'angle':nn.Parameter(torch.zeros(E))}

def fit(method,h,y,target,seed):
 p=init(method,seed);opt=torch.optim.Adam(list(p.values()),lr=.025);t=time.perf_counter()
 for _ in range(STEPS):
  opt.zero_grad();ww=weights(method,p);pred=torch.einsum('nd,edh->neh',h,ww)[torch.arange(len(y)),y];loss=(pred-target).square().mean();loss.backward();opt.step()
 return {k:v.detach() for k,v in p.items()},time.perf_counter()-t

def pack(method,p,router):
 vals=[]
 for k in sorted(p):vals.append(p[k].contiguous().reshape(-1))
 vals += [v.contiguous().reshape(-1) for v in router.state_dict().values()]
 flat=torch.cat(vals).float().numpy().tobytes();meta=json.dumps({'method':method,'shapes':{k:list(v.shape) for k,v in p.items()},'router_shapes':{k:list(v.shape) for k,v in router.state_dict().items()},'experts':E},sort_keys=True,separators=(',',':')).encode();return b'MA274\0'+struct.pack('<I',len(meta))+meta+flat

def run(phase):
 rows=[]
 for world in (DEV if phase=='development' else FRESH):
  for seed in SEEDS:
   (xt,ht,yt),(xv,hv,yv),teacher=make(world,seed);targets=torch.einsum('nd,edh->neh',ht,teacher)[torch.arange(len(yt)),yt]
   router=router_fit(xt[:,:2],yt,world+seed);routed=router(xv[:,:2]).argmax(-1);racc=float((routed==yv).float().mean())
   for method in ['standard_moe','tied','ia3_gate','rank1','givens_mirror','boft_mirror','independent_upper']:
    if method=='independent_upper':p={'W':teacher};sec=0.
    else:p,sec=fit(method,ht,yt,targets,world*31+seed)
    ww=teacher if method=='independent_upper' else weights(method,p);allp=torch.einsum('nd,edh->neh',hv,ww);oracle=allp[torch.arange(len(yv)),yv];pred=allp[torch.arange(len(yv)),routed]
    metric=lambda z:float((z-teacher_eval).square().mean().sqrt()/teacher_eval.square().mean().sqrt())
    teacher_eval=torch.einsum('nd,edh->neh',hv,teacher)[torch.arange(len(yv)),yv]
    blob=pack(method,p,router);path=ROOT/'artifacts'/'payloads'/f'{world}_{seed}_{method}.bin';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
    rows.append({'phase':phase,'world':world,'seed':seed,'method':method,'routed_nrmse':metric(pred),'oracle_route_nrmse':metric(oracle),'router_accuracy':racc,'payload_bytes':len(blob),'active_macs':len(yv)*D*D,'train_seconds':sec,'sha256':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
