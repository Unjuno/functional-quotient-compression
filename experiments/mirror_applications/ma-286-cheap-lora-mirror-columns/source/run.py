"""MA-286: fixed column selectors versus shared low-rank View codes."""
import argparse,csv,hashlib,json,math,struct,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];D,R,T=32,4,8;DEV,FRESH=[28600,28601],[28610,28611,28612];SEEDS=[0,1,2];UPDATES=400
torch.set_num_threads(2)
def make(world,seed):
 g=torch.Generator().manual_seed(world*100003+seed*7919+286);w=torch.randn(D,D,generator=g)/math.sqrt(D);x=torch.randn(256,D,generator=g);xv=torch.randn(512,D,generator=g)
 # Shared orthonormal column subspace and task coefficients.
 q,_=torch.linalg.qr(torch.randn(D,R,generator=g));right=torch.randn(R,D,generator=g)*.1;coeff=torch.randn(T,R,generator=g)*.15;aligned=torch.stack([q@torch.diag(coeff[t])@right for t in range(T)])
 # Independent rank-R updates.
 independent=torch.stack([torch.randn(D,R,generator=g)@torch.randn(R,D,generator=g)*.1 for _ in range(T)])
 return w,x,xv,q,right,coeff,aligned,independent
def pack(parts,method):
 flat=torch.cat([z.flatten() for z in parts]).float().numpy().tobytes();meta=json.dumps({'method':method,'shapes':[list(z.shape) for z in parts],'dim':D,'rank':R},sort_keys=True,separators=(',',':')).encode();return b'MA286\0'+struct.pack('<I',len(meta))+meta+flat
def nrmse(a,b):return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-12))
def run(phase):
 rows=[]
 for world in (DEV if phase=='development' else FRESH):
  for seed in SEEDS:
   w,x,xv,q,right,coeff,aligned,ind=make(world,seed)
   for stratum,targets in [('shared_subspace',aligned),('independent',ind)]:
    y=torch.einsum('nd,tdh->nth',x,targets)
    for method in ['none','fixed_cla','cheap_lora','shared_coeff','mirror_address','full_lora']:
     start=time.perf_counter();base=w
     if method=='none':delta=torch.zeros_like(targets);parts=[w]
     elif method=='fixed_cla':
      fixed=torch.eye(D)[:,:R];fixed_right=right;a=nn.Parameter(torch.zeros(T,R));opt=torch.optim.Adam([a],lr=.04)
      for _ in range(UPDATES):opt.zero_grad();delta=torch.stack([fixed@torch.diag(a[t])@fixed_right for t in range(T)]);pred=torch.einsum('nd,tdh->nth',x,delta);loss=(pred-y).square().mean();loss.backward();opt.step()
      delta=torch.stack([fixed@torch.diag(a.detach()[t])@fixed_right for t in range(T)]);parts=[w,fixed,fixed_right,a.detach()]
     elif method in ('shared_coeff','mirror_address'):
      a=nn.Parameter(torch.zeros(T,R));opt=torch.optim.Adam([a],lr=.04)
      for _ in range(UPDATES):opt.zero_grad();delta=torch.stack([q@torch.diag(a[t])@right for t in range(T)]);pred=torch.einsum('nd,tdh->nth',x,delta);loss=(pred-y).square().mean();loss.backward();opt.step()
      a=a.detach();delta=torch.stack([q@torch.diag(a[t])@right for t in range(T)]);parts=[w,q,right,a]
     else:
      aa=nn.Parameter(torch.randn(T,D,R)*.01);bb=nn.Parameter(torch.zeros(T,R,D));opt=torch.optim.Adam([aa,bb],lr=.04)
      for _ in range(UPDATES):opt.zero_grad();delta=aa@bb;pred=torch.einsum('nd,tdh->nth',x,delta);loss=(pred-y).square().mean();loss.backward();opt.step()
      aa=aa.detach();bb=bb.detach();delta=aa@bb;parts=[w,aa,bb]
     sec=time.perf_counter()-start;pred=torch.einsum('nd,tdh->nth',xv,delta);truth=torch.einsum('nd,tdh->nth',xv,targets);err=nrmse(pred,truth);inter=float((pred-truth).square().mean().sqrt());blob=pack(parts,method);path=ROOT/'artifacts'/'payloads'/f'{world}_{seed}_{stratum}_{method}.bin';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
     rows.append({'phase':phase,'world':world,'seed':seed,'stratum':stratum,'method':method,'nrmse':err,'payload_bytes':len(blob),'fit_seconds':sec,'interference':inter,'sha256':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=rows[0]);wri.writeheader();wri.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
