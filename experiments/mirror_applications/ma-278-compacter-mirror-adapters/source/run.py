"""MA-278: Compacter-style shared atom bank and low-description task codes."""
import argparse,csv,hashlib,json,math,struct,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];D,T,A=16,8,4;DEV,FRESH=[27800,27801],[27810,27811,27812];SEEDS=[0,1,2];UPDATES=400
torch.set_num_threads(2)

def atoms(g):
 # Compacter-like Kronecker atoms: slow 4x4 factors ⊗ fast 4x4 rank-one factors, cropped to 16x16.
 slow=torch.randn(A,4,4,generator=g);fast=torch.randn(A,4,4,generator=g)
 return torch.stack([torch.kron(slow[i],fast[i]) for i in range(A)]),slow,fast
def make(world,seed):
 g=torch.Generator().manual_seed(world*100003+seed*7919+278);w=torch.randn(D,D,generator=g)/math.sqrt(D);x=torch.randn(256,D,generator=g);v=torch.randn(512,D,generator=g);basis,slow,fast=atoms(g)
 direction=torch.ones(A)/math.sqrt(A);coeff=torch.linspace(-.45,.45,T)[:,None];aligned=coeff[:,:,None,None]*(direction[None,:,None,None]*basis[None]).sum(1)[:,None]
 aligned=aligned[:,0]
 independent=torch.stack([torch.randn(D,2,generator=g)@torch.randn(2,D,generator=g)*.12 for _ in range(T)])
 return w,x,v,basis,slow,fast,aligned,independent
def nrmse(a,b):return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-12))
def pack(payload,method,shapes):
 flat=torch.cat([v.flatten() for v in payload]).float().numpy().tobytes();meta=json.dumps({'method':method,'shapes':shapes},sort_keys=True,separators=(',',':')).encode();return b'MA278\0'+struct.pack('<I',len(meta))+meta+flat
def run(phase):
 rows=[]
 for world in (DEV if phase=='development' else FRESH):
  for seed in SEEDS:
   w,x,xv,basis,slow,fast,aligned,ind=make(world,seed)
   for stratum,targets in [('shared_atoms',aligned),('independent',ind)]:
    y=torch.einsum('nd,tdh->nth',x,w[None]+targets)
    for method in ['none','compacter','mirror_scalar','generic_coeff','lora2']:
     start=time.perf_counter()
     if method=='none':delta=torch.zeros_like(targets);code=torch.empty(0);payload=[w];shapes=[list(w.shape)]
     elif method=='compacter':
      # native per-task fast coefficients over the shared Compacter atoms
      code=nn.Parameter(torch.zeros(T,A));opt=torch.optim.Adam([code],lr=.06)
      for _ in range(UPDATES):opt.zero_grad();delta=torch.einsum('ta,aij->tij',code,basis);pred=torch.einsum('nd,tdh->nth',x,w[None]+delta);loss=(pred-y).square().mean();loss.backward();opt.step()
      delta=torch.einsum('ta,aij->tij',code.detach(),basis);code=code.detach();payload=[w,slow,fast,code];shapes=[list(z.shape) for z in payload]
     elif method in ('mirror_scalar','generic_coeff'):
      # One shared atom direction with a scalar per task, intentionally a generic additive coefficient control.
      direction=torch.ones(A)/math.sqrt(A);feat=torch.einsum('a,aij->ij',direction,basis);code=nn.Parameter(torch.zeros(T));opt=torch.optim.Adam([code],lr=.06)
      for _ in range(UPDATES):opt.zero_grad();delta=code[:,None,None]*feat;pred=torch.einsum('nd,tdh->nth',x,w[None]+delta);loss=(pred-y).square().mean();loss.backward();opt.step()
      delta=code.detach()[:,None,None]*feat;code=code.detach();payload=[w,slow,fast,code];shapes=[list(z.shape) for z in payload]
     else:
      # Independent rank-2 update per task.
      aa=nn.Parameter(torch.randn(T,D,2)*.01);bb=nn.Parameter(torch.zeros(T,2,D));opt=torch.optim.Adam([aa,bb],lr=.05)
      for _ in range(UPDATES):opt.zero_grad();delta=aa@bb;pred=torch.einsum('nd,tdh->nth',x,w[None]+delta);loss=(pred-y).square().mean();loss.backward();opt.step()
      delta=aa.detach()@bb.detach();code=torch.cat([aa.detach().flatten(),bb.detach().flatten()]);payload=[w,aa.detach(),bb.detach()];shapes=[list(w.shape),list(aa.shape),list(bb.shape)]
     sec=time.perf_counter()-start
     pred=torch.einsum('nd,tdh->nth',xv,w[None]+delta);target=torch.einsum('nd,tdh->nth',xv,w[None]+targets);errs=torch.linalg.vector_norm(pred-target,dim=(0,2))/torch.linalg.vector_norm(target,dim=(0,2)).clamp_min(1e-12)
     inter=float((pred-target).square().mean().sqrt());blob=pack(payload,method,shapes);path=ROOT/'artifacts'/'payloads'/f'{world}_{seed}_{stratum}_{method}.bin';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
     rows.append({'phase':phase,'world':world,'seed':seed,'stratum':stratum,'task':-1,'method':method,'nrmse':float(errs.mean()),'payload_bytes':len(blob),'interference':inter,'train_seconds':sec,'sha256':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=rows[0]);wri.writeheader();wri.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
