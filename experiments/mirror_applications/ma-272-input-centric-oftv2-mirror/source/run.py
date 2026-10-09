"""MA-272: input-centric orthogonal transforms versus materialized weight views."""
import argparse,csv,hashlib,json,math,struct,time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];D,TASKS=32,8;DEV,FRESH=[27200,27201],[27210,27211,27212];SEEDS=[0,1,2]
torch.set_num_threads(2)

def givens(theta):
 q=torch.eye(D);c,s=torch.cos(theta),torch.sin(theta);q[0,0]=c;q[0,1]=-s;q[1,0]=s;q[1,1]=c;return q

def make(world,seed):
 g=torch.Generator().manual_seed(world*100003+seed*7919+272);w=torch.randn(D,D,generator=g)/math.sqrt(D)
 x=torch.randn(256,D,generator=g);a=torch.randn(1024,D,generator=g)
 angles=torch.linspace(-.7,.7,TASKS);mats=[]
 for _ in range(TASKS):
  z=torch.randn(D,D,generator=g);u,r=torch.linalg.qr(z);sg=torch.where(torch.diag(r)<0,-1.,1.);mats.append(u@torch.diag(sg))
 return w,x,a,angles,torch.stack(mats)

def apply_input(x,w,q): return (x@q)@w
def materialize(w,q): return q@w
def payload(w,method,code):
 b=torch.cat([w.flatten(),code.flatten()]).float().numpy().tobytes();meta=json.dumps({'method':method,'weight_shape':list(w.shape),'code_shape':list(code.shape)},sort_keys=True,separators=(',',':')).encode();return b'MA272\0'+struct.pack('<I',len(meta))+meta+b
def relerr(a,b):return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-12))
def timed(fn,reps=100):
 for _ in range(5):fn()
 t=time.perf_counter()
 for _ in range(reps):fn()
 return (time.perf_counter()-t)/reps

def run(phase):
 rows=[]
 for world in (DEV if phase=='development' else FRESH):
  for seed in SEEDS:
   w,x,a,angles,ind=make(world,seed)
   for stratum in ['aligned_plane','independent_orthogonal']:
    for task in range(TASKS):
     q=givens(angles[task]) if stratum=='aligned_plane' else ind[task]
     methods=[('oftv2_input',q,q),('oft_materialized',q,q)]
     if stratum=='aligned_plane':methods += [('mirror_input_angle',torch.tensor([float(angles[task])]),q),('generic_plane_scalar',torch.tensor([float(angles[task])]),q)]
     methods += [('shared_baseline',torch.empty(0),torch.eye(D))]
     ya=apply_input(a,w,q)
     for name,code,qmat in methods:
      if name=='oft_materialized':
       t=time.perf_counter();wm=materialize(w,qmat);matsec=time.perf_counter()-t;pred=a@wm
      elif name=='shared_baseline':matsec=0.;pred=a@w
      else:matsec=0.;pred=apply_input(a,w,qmat)
      err=relerr(pred,ya)
      if name=='oftv2_input':
       fwd=timed(lambda:apply_input(a,w,qmat));extra_macs=D*len(a)*D
      elif name=='oft_materialized':
       wm=materialize(w,qmat);fwd=timed(lambda:a@wm);extra_macs=0
      elif name=='shared_baseline':fwd=timed(lambda:a@w);extra_macs=0
      else:fwd=timed(lambda:apply_input(a,w,qmat));extra_macs=D*len(a)*D
      # Both OFT forms store the same source transform code. Materialized weights
      # are deterministic from the already-paid shared W and Q, so are not charged twice.
      stored_code=(torch.tensor([float(angles[task])]) if name in ('mirror_input_angle','generic_plane_scalar') else (torch.empty(0) if name=='shared_baseline' else q))
      blob=payload(w,name,stored_code);path=ROOT/'artifacts'/'payloads'/f'{world}_{seed}_{stratum}_{task}_{name}.bin';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
      rows.append({'phase':phase,'world':world,'seed':seed,'stratum':stratum,'task':task,'method':name,'relative_function_error':err,'payload_bytes':len(blob),'materialize_seconds':matsec,'forward_seconds':fwd,'extra_macs':extra_macs,'sha256':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=rows[0]);wri.writeheader();wri.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
