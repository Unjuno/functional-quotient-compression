"""MA-282: a simple Monarch block-diagonal/permutation factor screen."""
import argparse,csv,hashlib,json,math,struct,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];D,T,B=16,8,4;DEV,FRESH=[28200,28201],[28210,28211,28212];SEEDS=[0,1,2];UPDATES=400
torch.set_num_threads(2)
PERM=torch.roll(torch.arange(D),B)
def monarch(a,b):
 # M = blockdiag(a) P blockdiag(b), with four 4x4 blocks on each side.
 aa=torch.block_diag(*[a[i] for i in range(D//B)]);bb=torch.block_diag(*[b[i] for i in range(D//B)]);return aa[:,PERM]@bb
def make(world,seed):
 g=torch.Generator().manual_seed(world*100003+seed*7919+282);w=torch.randn(D,D,generator=g)/math.sqrt(D);x=torch.randn(256,D,generator=g);xv=torch.randn(512,D,generator=g)
 a=torch.randn(D//B,B,B,generator=g)*.1;b=torch.randn(D//B,B,B,generator=g)*.1;direction=torch.randn(D//B,B,B,generator=g)*.2
 w=w+monarch(a,b)
 coeff=torch.linspace(-.5,.5,T);aligned=torch.stack([c*monarch(direction,torch.eye(B).repeat(D//B,1,1)) for c in coeff])
 independent=torch.stack([torch.randn(D,D,generator=g)*.1 for _ in range(T)])
 return w,x,xv,a,b,direction,aligned,independent
def pack(parts,method):
 flat=torch.cat([p.flatten() for p in parts]).float().numpy().tobytes();meta=json.dumps({'method':method,'shapes':[list(p.shape) for p in parts],'block':B,'permutation':PERM.tolist()},sort_keys=True,separators=(',',':')).encode();return b'MA282\0'+struct.pack('<I',len(meta))+meta+flat
def nrmse(p,y):return float(torch.linalg.vector_norm(p-y)/torch.linalg.vector_norm(y).clamp_min(1e-12))
def run(phase):
 rows=[]
 for world in (DEV if phase=='development' else FRESH):
  for seed in SEEDS:
   w,x,xv,a,b,direction,aligned,ind=make(world,seed)
   for stratum,targets in [('aligned_monarch',aligned),('independent',ind)]:
    for method in ['shared','monarch_native','mirror_scalar','generic_scalar','lora2','dense_upper']:
     y=torch.einsum('nd,tdh->nth',x,w[None]+targets);start=time.perf_counter()
     if method=='shared':delta=torch.zeros_like(targets);parts=[w]
     elif method in ('mirror_scalar','generic_scalar'):
      feat=monarch(direction,torch.eye(B).repeat(D//B,1,1));xf=x@feat;res=y-torch.einsum('nd,tdh->nth',x,w[None]);code=(xf[:,None,:]*res).sum((0,2))/(xf.square().sum()+1e-12);delta=code[:,None,None]*feat;parts=[w,a,b,direction,code]
     elif method=='monarch_native':
      code=nn.Parameter(torch.zeros(T,D//B,B,B));opt=torch.optim.Adam([code],lr=.04)
      for _ in range(UPDATES):opt.zero_grad();delta=torch.stack([monarch(code[t],torch.eye(B).repeat(D//B,1,1)) for t in range(T)]);pred=torch.einsum('nd,tdh->nth',x,w[None]+delta);loss=(pred-y).square().mean();loss.backward();opt.step()
      code=code.detach();delta=torch.stack([monarch(code[t],torch.eye(B).repeat(D//B,1,1)) for t in range(T)]);parts=[w,code]
     elif method=='lora2':
      aa=nn.Parameter(torch.randn(T,D,2)*.01);bb=nn.Parameter(torch.zeros(T,2,D));opt=torch.optim.Adam([aa,bb],lr=.04)
      for _ in range(UPDATES):opt.zero_grad();delta=aa@bb;pred=torch.einsum('nd,tdh->nth',x,w[None]+delta);loss=(pred-y).square().mean();loss.backward();opt.step()
      aa=aa.detach();bb=bb.detach();delta=aa@bb;parts=[w,aa,bb]
     else:delta=targets;parts=[w,targets]
     sec=time.perf_counter()-start;pred=torch.einsum('nd,tdh->nth',xv,w[None]+delta);truth=torch.einsum('nd,tdh->nth',xv,w[None]+targets);err=nrmse(pred,truth);blob=pack(parts,method);path=ROOT/'artifacts'/'payloads'/f'{world}_{seed}_{stratum}_{method}.bin';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
     t=time.perf_counter();_=torch.einsum('nd,tdh->nth',xv,w[None]+delta);apply=time.perf_counter()-t
     rows.append({'phase':phase,'world':world,'seed':seed,'stratum':stratum,'method':method,'nrmse':err,'payload_bytes':len(blob),'fit_seconds':sec,'apply_seconds':apply,'sha256':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=rows[0]);wri.writeheader();wri.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
