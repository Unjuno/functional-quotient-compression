"""MA-271: native OFT task matrices versus compact orthogonal Mirror codes."""
import argparse,csv,hashlib,json,math,struct,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]; DIN,H,DOUT,TASKS=16,32,8,8
DEV,FRESH,SEEDS=[27100,27101],[27110,27111,27112],[0,1,2]
torch.set_num_threads(2)

def world_data(world,seed):
 g=torch.Generator().manual_seed(world*100003+seed*7919+271)
 w1=torch.randn(DIN,H,generator=g)/math.sqrt(DIN);w2=torch.randn(H,DOUT,generator=g)/math.sqrt(H)
 def samples(n,off):
  q=torch.Generator().manual_seed(world*100003+seed*7919+off);x=torch.randn(n,DIN,generator=q);return F.relu(x@w1+0.8)
 hs,hv=samples(128,19),samples(512,23)
 angles=torch.linspace(-.65,.65,TASKS)
 # Independent orthogonal transforms are sampled once per task/world/seed.
 mats=[]
 for _ in range(TASKS):
  z=torch.randn(DOUT,DOUT,generator=g);q,r=torch.linalg.qr(z);sg=torch.where(torch.diag(r)<0,-1.,1.);mats.append(q@torch.diag(sg))
 return w1,w2,hs,hv,angles,torch.stack(mats)

def plane(theta):
 r=torch.eye(DOUT);c,s=torch.cos(theta),torch.sin(theta);r[0,0]=c;r[0,1]=-s;r[1,0]=s;r[1,1]=c;return r

def orthogonal_fit(x,y,steps=220,lr=.04):
 # Cayley parameterization keeps the learned OFT matrix orthogonal.
 a=nn.Parameter(torch.zeros(DOUT,DOUT));opt=torch.optim.Adam([a],lr=lr);t=time.perf_counter()
 for _ in range(steps):
  opt.zero_grad();skew=a-a.T;eye=torch.eye(DOUT);q=torch.linalg.solve(eye+skew,eye-skew);loss=(x@q-y).square().mean();loss.backward();opt.step()
 with torch.no_grad():
  skew=a-a.T;q=torch.linalg.solve(torch.eye(DOUT)+skew,torch.eye(DOUT)-skew)
 return q.detach(),time.perf_counter()-t,steps

def angle_fit(x,y):
 t=time.perf_counter();grid=torch.linspace(-1.3,1.3,1001);pred=torch.einsum('nd,gde->nge',x,torch.stack([plane(a) for a in grid])) # N,G,D
 loss=(pred-y[:,None,:]).square().mean((0,2));idx=int(loss.argmin());lo=float(grid[max(idx-1,0)]);hi=float(grid[min(idx+1,len(grid)-1)])
 phi=(math.sqrt(5)-1)/2
 def f(a):return float((x@plane(torch.tensor(a))-y).square().mean())
 c=hi-phi*(hi-lo);d=lo+phi*(hi-lo);fc,fd=f(c),f(d)
 for _ in range(45):
  if fc<fd:hi,d,fd=d,c,fc;c=hi-phi*(hi-lo);fc=f(c)
  else:lo,c,fc=c,d,fd;d=lo+phi*(hi-lo);fd=f(d)
 return (lo+hi)/2,time.perf_counter()-t

def nrmse(a,b):return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-12))
def payload(shared,method,codes):
 vals=torch.cat([shared.reshape(-1).float(),codes.reshape(-1).float()]).numpy().tobytes();meta=json.dumps({'method':method,'shared':list(shared.shape),'codes':list(codes.shape),'dtype':'float32'},sort_keys=True,separators=(',',':')).encode();return b'MA271\0'+struct.pack('<I',len(meta))+meta+vals

def run(phase):
 rows=[]
 for world in (DEV if phase=='development' else FRESH):
  for seed in SEEDS:
   w1,w2,hs,hv,angles,independent=world_data(world,seed);shared=torch.cat([w1.flatten(),w2.flatten()]);hs=hs@w2;hv=hv@w2
   for stratum in ['aligned_plane','independent_orthogonal']:
    for task in range(TASKS):
     qtrue=plane(angles[task]) if stratum=='aligned_plane' else independent[task]
     yt=hs@qtrue;ya=hv@qtrue
     methods=[]
     # Native full OFT matrix, adapted from support pairs.
     q,sec,updates=orthogonal_fit(hs,yt);methods.append(('oft_independent',q,q,sec,updates))
     # Orthogonal Procrustes upper control fitted directly to the known teacher outputs.
     u,_,vh=torch.linalg.svd(hs.T@yt);q_oracle=u@vh;methods.append(('oft_oracle_upper',q_oracle,q_oracle,0.,0))
     if stratum=='aligned_plane':
      a,sec=angle_fit(hs,yt);qm=plane(torch.tensor(a));methods.append(('mirror_angle',torch.tensor(a).reshape(1),qm,sec,0))
      # Same coordinate family serves as the simplest non-Mirror coefficient control.
      methods.append(('generic_plane_scalar',torch.tensor(a).reshape(1),qm,sec,0))
     # Shared orthogonal map: fit identity/common function, no per-task view.
     eye=torch.eye(DOUT);methods.append(('shared_identity',eye,eye,0.,0))
     for name,code,qhat,fitsec,updates in methods:
      pred=hv@qhat;blob=payload(shared,name,code);path=ROOT/'artifacts'/'payloads'/f'{world}_{seed}_{stratum}_{task}_{name}.bin';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
      t=time.perf_counter();_=hv@qhat;apply=time.perf_counter()-t
      # geometry retention compares all pairwise distances after transform.
      gram=torch.cdist(hv,hv);gout=torch.cdist(pred,pred);geom=float((gram-gout).abs().mean()/gram.mean().clamp_min(1e-12))
      rows.append({'phase':phase,'world':world,'seed':seed,'stratum':stratum,'task':task,'method':name,'nrmse':nrmse(pred,ya),'geometry_distortion':geom,'payload_bytes':len(blob),'fit_seconds':fitsec,'apply_seconds':apply,'updates':updates,'sha256':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
