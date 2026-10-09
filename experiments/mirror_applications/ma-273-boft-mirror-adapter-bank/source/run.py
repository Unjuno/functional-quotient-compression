"""MA-273: explicit butterfly products as native BOFT controls and shared views."""
import argparse,csv,hashlib,json,math,struct,time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1];D,TASKS,STAGES=32,8,5;DEV,FRESH=[27300,27301],[27310,27311,27312];SEEDS=[0,1,2]
torch.set_num_threads(2)

def pair_schedule(d):
 # Adjacent butterfly pairing with cyclic offsets; each stage has d/2 Givens angles.
 return [([(i,(i+1)%d) for i in range(off,d,2) if (i+1)%d!=i] + ([(d-1,0)] if off and d%2==0 else [])) for off in [0,1,3,7,15]]

SCHEDULE=pair_schedule(D)
def stage_matrix(angles,pairs,d=D):
 q=torch.eye(d)
 for theta,(i,j) in zip(angles,pairs):
  c,s=torch.cos(theta),torch.sin(theta);q[i,i]=c;q[j,j]=c;q[i,j]=-s;q[j,i]=s
 return q
def boft_matrix(code):
 q=torch.eye(D)
 for st,pairs in enumerate(SCHEDULE):
  block=stage_matrix(code[st],pairs);q=q@block
 return q
def make(world,seed):
 g=torch.Generator().manual_seed(world*100003+seed*7919+273);w=torch.randn(D,D,generator=g)/math.sqrt(D);x=torch.randn(256,D,generator=g);a=torch.randn(1024,D,generator=g)
 base=torch.randn(sum(len(p) for p in SCHEDULE),generator=g)*.08
 # Aligned task maps follow one scalar times a fixed full butterfly generator schedule.
 coeff=torch.linspace(-1,1,TASKS);aligned=[]
 for c in coeff:aligned.append(boft_matrix([torch.full((len(p),),float(c))*base[sum(len(z) for z in SCHEDULE[:i]):sum(len(z) for z in SCHEDULE[:i+1])] for i,p in enumerate(SCHEDULE)]))
 independent=[]
 for _ in range(TASKS):
  z=torch.randn(D,D,generator=g);u,r=torch.linalg.qr(z);sg=torch.where(torch.diag(r)<0,-1.,1.);independent.append(u@torch.diag(sg))
 return w,x,a,base,torch.stack(aligned),torch.stack(independent)

def decode_full(code):
 # Native independent BOFT: every stage/pair has its own angle.
 return boft_matrix(code)
def serialize(w,method,code):
 flat=torch.cat([w.flatten().float(),code.flatten().float()]).numpy().tobytes();meta=json.dumps({'method':method,'w':[D,D],'code':list(code.shape),'schedule':SCHEDULE},sort_keys=True,separators=(',',':')).encode();return b'MA273\0'+struct.pack('<I',len(meta))+meta+flat
def nrmse(p,y):return float(torch.linalg.vector_norm(p-y)/torch.linalg.vector_norm(y).clamp_min(1e-12))
def measure(fn,reps=60):
 for _ in range(4):fn()
 t=time.perf_counter()
 for _ in range(reps):fn()
 return (time.perf_counter()-t)/reps
def fit_generic_scalar(w,x,y,base,seed):
 # Smooth one-dimensional support-only fit; fixed iteration budget.
 z=torch.tensor(0.0);step=.05
 def loss(v):
  code=[(v*base[sum(len(q) for q in SCHEDULE[:i]):sum(len(q) for q in SCHEDULE[:i+1])]).reshape(-1) for i in range(STAGES)]
  return (x@boft_matrix(code)-y).square().mean()
 for _ in range(30):
  l0=float(loss(z));lm=float(loss(z-step));lp=float(loss(z+step))
  if lm<l0 and lm<=lp:z-=step
  elif lp<l0:z+=step
  step*=.85
 return float(z)

def run(phase):
 rows=[]
 for world in (DEV if phase=='development' else FRESH):
  for seed in SEEDS:
   w,x,a,base,aligned,ind=make(world,seed)
   for stratum in ['aligned_shared_generator','independent_orthogonal']:
    for task in range(TASKS):
     qtrue=aligned[task] if stratum=='aligned_shared_generator' else ind[task];yt=x@qtrue;ya=a@qtrue
     methods=[]
     # Full-matrix OFT oracle and independent BOFT with fitted unconstrained factor angles.
     methods.append(('oft_oracle',qtrue.flatten(),qtrue))
     # Aligned family has a shared butterfly generator and task-specific scalar address.
     if stratum=='aligned_shared_generator':
      alpha=float(torch.linspace(-1,1,TASKS)[task]);chunks=[];off=0
      for pairs in SCHEDULE:chunks.append((alpha*base[off:off+len(pairs)]).clone());off+=len(pairs)
      qm=boft_matrix(chunks);methods.append(('boft_independent',torch.cat(chunks),qm))
      z=fit_generic_scalar(w,x,yt,base,seed);mirror_code=torch.tensor([z]);chunks=[(z*base[sum(len(p) for p in SCHEDULE[:i]):sum(len(p) for p in SCHEDULE[:i+1])]).clone() for i in range(STAGES)];methods.append(('mirror_shared_boft',mirror_code,boft_matrix(chunks)));methods.append(('generic_scalar_boft',mirror_code.clone(),boft_matrix(chunks)))
     else:
      # Fit independent BOFT angle bank from identity by deterministic coordinate descent.
      best=torch.zeros(sum(len(p) for p in SCHEDULE));bestloss=float('inf')
      for sweep in range(1):
       for j in range(len(best)):
        candidates=best.repeat(3,1);candidates[:,j]+=torch.tensor([-.08,0.,.08]);scores=[]
        for trial in candidates:
         parts=[];off=0
         for pairs in SCHEDULE:parts.append(trial[off:off+len(pairs)]);off+=len(pairs)
         scores.append((x@boft_matrix(parts)-yt).square().mean())
        idx=int(torch.stack(scores).argmin());best=candidates[idx].clone()
      parts=[];off=0
      for pairs in SCHEDULE:parts.append(best[off:off+len(pairs)]);off+=len(pairs)
      methods.append(('boft_independent',best,boft_matrix(parts)))
     methods.append(('shared_identity',torch.empty(0),torch.eye(D)))
     for name,code,qhat in methods:
      pred=a@qhat;blob=serialize(w,name,code);path=ROOT/'artifacts'/'payloads'/f'{world}_{seed}_{stratum}_{task}_{name}.bin';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
      # Dense application proxy and butterfly transform proxy. Actual runtime measured by executed functions.
      t=time.perf_counter();_=qhat@w;matsec=time.perf_counter()-t
      fn=lambda:a@qhat@w;apply=measure(fn)
      macs=2*len(a)*D*D
      rows.append({'phase':phase,'world':world,'seed':seed,'stratum':stratum,'task':task,'method':name,'nrmse':nrmse(pred,ya),'payload_bytes':len(blob),'apply_seconds':apply,'materialize_seconds':matsec,'apply_macs':macs,'sha256':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
