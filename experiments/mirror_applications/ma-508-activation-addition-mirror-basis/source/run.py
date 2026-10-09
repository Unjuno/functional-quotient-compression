#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,math,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads'
D,R,N=64,4,64;DEV=[50800,50801];FRESH=[50820,50821,50822];SEEDS=[0,1,2];METHODS=['explicit','coeff_fp32','coeff_fp16','mirror_fp16','mirror_private_fp16']
torch.set_num_threads(1)
G=torch.Generator().manual_seed(5080001);B=torch.linalg.qr(torch.randn(D,R,generator=G)).Q.contiguous();Z=torch.randn(R,generator=G);Z=Z/Z.norm()

def rotate(z,a):
 out=z.expand(len(a),-1).clone();x,y=out[:,0].clone(),out[:,1].clone();c,s=a[:,0].cos(),a[:,0].sin();out[:,0]=c*x-s*y;out[:,1]=s*x+c*y
 x,y=out[:,2].clone(),out[:,3].clone();c,s=a[:,1].cos(),a[:,1].sin();out[:,2]=c*x-s*y;out[:,3]=s*x+c*y
 return out

def make_bank(world,seed,rho):
 g=torch.Generator().manual_seed(world*100003+seed*7919+508)
 angles=(torch.rand(N,2,generator=g)-.5)*2*math.pi
 coeff=rotate(Z,angles)
 if rho: coeff=coeff+rho*torch.randn(N,R,generator=g)
 vectors=coeff@B.T
 probes=[]
 for i in range(N):
  pg=torch.Generator().manual_seed(world*97+seed*1009+i*13+811)
  v=torch.randn(32,D,generator=pg);v= torch.nn.functional.normalize(v,dim=-1);probes.append(v)
 probes=torch.stack(probes)
 return coeff,vectors,angles,probes

def recover_angles(coeff):
 out=[]
 for c in coeff:
  a0=torch.atan2(Z[0]*c[1]-Z[1]*c[0],Z[0]*c[0]+Z[1]*c[1])
  a1=torch.atan2(Z[2]*c[3]-Z[3]*c[2],Z[2]*c[2]+Z[3]*c[3])
  out.append(torch.stack((a0,a1)))
 return torch.stack(out)

def pack(method,coeff,vectors):
 if method=='explicit': state={'method':method,'vectors':vectors}
 elif method=='coeff_fp32': state={'method':method,'basis':B,'coefficients':coeff}
 elif method=='coeff_fp16': state={'method':method,'basis':B,'coefficients':coeff.half()}
 else:
  ang=recover_angles(coeff);base={'method':method,'basis':B,'seed':Z,'angles':ang.half()}
  if method=='mirror_private_fp16':
   recon=rotate(Z,ang)@B.T;res=(vectors-recon)@B;state={**base,'private_residual':res.half()}
  else:state=base
 s=io.BytesIO();torch.save(state,s);return s.getvalue()

def decode(state):
 m=state['method']
 if m=='explicit':return state['vectors']
 if m.startswith('coeff_'):return state['coefficients'].float()@state['basis'].T
 c=rotate(state['seed'],state['angles'].float())
 if 'private_residual' in state:c=c+state['private_residual'].float()
 return c@state['basis'].T

def evaluate(method,payload,coeff,vectors,probes,phase,world,seed,rho):
 st=torch.load(io.BytesIO(payload),weights_only=False);rec=decode(st);err=rec-vectors
 nrm=float(err.norm()/(vectors.norm()+1e-12));target=(vectors* vectors).sum(-1)+1e-12;eff=((rec*vectors).sum(-1)/target).mean().item()
 drift=[]
 for i in range(N): drift.extend((probes[i]@err[i]).tolist())
 off=float(torch.tensor(drift).square().mean().sqrt())
 # Batch decode timing includes angle transform and basis application.
 for _ in range(5):decode(st)
 t=time.perf_counter()
 for _ in range(100):decode(st)
 dt=(time.perf_counter()-t)/100
 mac={'explicit':0,'coeff_fp32':D*R,'coeff_fp16':D*R,'mirror_fp16':D*R+8,'mirror_private_fp16':D*R+8}[method]
 digest=hashlib.sha256(payload).hexdigest();path=PAY/phase/str(world)/str(seed)/f'rho{rho:.1f}'/(method+'.pt');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
 replay=torch.load(io.BytesIO(payload),weights_only=False);delta=float((decode(replay)-rec).abs().max())
 return {'phase':phase,'world':world,'seed':seed,'rho':rho,'method':method,'vector_nrmse':nrm,'efficacy_ratio':eff,'off_target_readout_drift':off,'private_fraction':(1.0 if 'private_residual' in st else 0.0),'serialized_payload_bytes':len(payload),'bytes_per_behavior':len(payload)/N,'application_mac_proxy_per_vector':mac,'decode_wall_seconds_bank':dt,'payload_sha256':digest,'max_replay_abs_error':delta,'payload_path':str(path.relative_to(ROOT.parents[2]))}

def run(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True);worlds=DEV if phase=='development' else FRESH;rows=[]
 for rho in [0.0,0.1]:
  for world in worlds:
   for seed in SEEDS:
    c,v,a,p=make_bank(world,seed,rho)
    for method in METHODS:
     rows.append(evaluate(method,pack(method,c,v),c,v,p,phase,world,seed,rho))
 dest=ART/('development.csv' if phase=='development' else 'fresh.csv')
 with dest.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 (ROOT/'RESULTS_CORE.csv').write_text('')
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 (ART/('development_runs.jsonl' if phase=='development' else 'fresh_runs.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
 print(json.dumps({'phase':phase,'rows':len(rows)},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);a=p.parse_args();run(a.phase)
