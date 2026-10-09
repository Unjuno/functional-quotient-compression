#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,math,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads'
D,R,CN,BN,CD=64,4,32,16,8;DEV=[50800,50801];FRESH=[50820,50821,50822];SEEDS=[0,1,2];THRESH=.35;METHODS=['cast_explicit','coeff_fp32','coeff_fp16','mirror_fp16','mirror_private_fp16']
torch.set_num_threads(1)
G=torch.Generator().manual_seed(5080001);B=torch.linalg.qr(torch.randn(D,R,generator=G)).Q.contiguous();Z=torch.randn(R,generator=G);Z=Z/Z.norm()

def rotate(z,a):
 out=z.expand(len(a),-1).clone();x,y=out[:,0].clone(),out[:,1].clone();c,s=a[:,0].cos(),a[:,0].sin();out[:,0]=c*x-s*y;out[:,1]=s*x+c*y
 x,y=out[:,2].clone(),out[:,3].clone();c,s=a[:,1].cos(),a[:,1].sin();out[:,2]=c*x-s*y;out[:,3]=s*x+c*y
 return out

def bank(world,seed,rho):
 g=torch.Generator().manual_seed(world*100003+seed*7919+510);cond=torch.nn.functional.normalize(torch.randn(CN,CD,generator=g),dim=-1);keys=torch.nn.functional.normalize(torch.randn(BN,CD,generator=g),dim=-1)
 ang=(torch.rand(BN,2,generator=g)-.5)*2*math.pi;coeff=rotate(Z,ang)
 if rho:coeff=coeff+rho*torch.randn(BN,R,generator=g)
 vec=coeff@B.T; sim=cond@keys.T;truth=sim>THRESH
 ng=torch.Generator().manual_seed(world*97+seed*1009+5101);noise=.2*torch.randn(CN,BN,CD,generator=ng);query=cond[:,None,:]+noise;query=torch.nn.functional.normalize(query,dim=-1);pred=torch.einsum('ijk,jk->ij',query,keys)>THRESH
 held=torch.tensor([[(i*7+j*11+seed)%5==0 for j in range(BN)] for i in range(CN)])
 pg=torch.Generator().manual_seed(world*13+seed*181+519);probes=torch.nn.functional.normalize(torch.randn(32,D,generator=pg),dim=-1)
 return cond,keys,coeff,vec,ang,truth,pred,held,probes

def extract_angles(coeff):
 out=[]
 for c in coeff:
  a=torch.atan2(Z[0]*c[1]-Z[1]*c[0],Z[0]*c[0]+Z[1]*c[1]);b=torch.atan2(Z[2]*c[3]-Z[3]*c[2],Z[2]*c[2]+Z[3]*c[3]);out.append(torch.stack((a,b)))
 return torch.stack(out)

def pack(method,cond,keys,coeff,vec):
 common={'method':method,'condition_prototypes':cond,'behavior_keys':keys,'trigger_threshold':THRESH,'behavior_count':BN}
 if method=='cast_explicit':state={**common,'vectors':vec}
 elif method=='coeff_fp32':state={**common,'basis':B,'coefficients':coeff}
 elif method=='coeff_fp16':state={**common,'basis':B,'coefficients':coeff.half()}
 else:
  a=extract_angles(coeff);state={**common,'basis':B,'seed':Z,'angles':a.half()}
  if method=='mirror_private_fp16':
   residual=(coeff-rotate(Z,a));state['private_residual']=residual.half()
 s=io.BytesIO();torch.save(state,s);return s.getvalue()

def decode(st):
 if st['method']=='cast_explicit':return st['vectors']
 if st['method'].startswith('coeff_'):return st['coefficients'].float()@st['basis'].T
 c=rotate(st['seed'],st['angles'].float())
 if 'private_residual' in st:c=c+st['private_residual'].float()
 return c@st['basis'].T

def evalone(method,payload,truth,pred,held,probes,phase,world,seed,rho):
 st=torch.load(io.BytesIO(payload),weights_only=False);vec=decode(st);target_vec=EVAL_TARGETS[(world,seed,rho)]
 sub=held;ii,jj=torch.nonzero(sub,as_tuple=True);yy=truth[sub];pp=pred[sub];neg=~yy;pos=yy
 fpr=float((pp&neg).sum()/(neg.sum()+1e-12));miss=float(((~pp)&pos).sum()/(pos.sum()+1e-12))
 applied=pp[:,None].float()*vec[jj]
 desired=yy[:,None].float()*target_vec[jj]
 if pos.any():
  efficacy=float(((applied[pos]*desired[pos]).sum(-1)/(desired[pos].square().sum(-1)+1e-12)).mean())
 else:efficacy=0.0
 false=neg&pp
 if false.any():
  false_beh=jj[false]
  off=float((vec[false_beh]@probes.T).square().mean().sqrt())
 else:off=0.0
 nrm=float((vec-target_vec).norm()/(target_vec.norm()+1e-12))
 for _ in range(3):decode(st)
 t=time.perf_counter()
 for _ in range(50):decode(st)
 dt=(time.perf_counter()-t)/50
 mac={'cast_explicit':0,'coeff_fp32':D*R,'coeff_fp16':D*R,'mirror_fp16':D*R+8,'mirror_private_fp16':D*R+8}[method]
 path=PAY/phase/str(world)/str(seed)/f'rho{rho:.1f}'/(method+'.pt');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload);digest=hashlib.sha256(payload).hexdigest();re=torch.load(io.BytesIO(payload),weights_only=False);rd=float((decode(re)-vec).abs().max())
 return {'phase':phase,'world':world,'seed':seed,'rho':rho,'method':method,'heldout_pairs':int(sub.sum()),'heldout_positive_pairs':int((sub&truth).sum()),'heldout_negative_pairs':int((sub&(~truth)).sum()),'false_trigger_rate':fpr,'miss_rate':miss,'on_target_efficacy':efficacy,'off_target_readout_drift':off,'behavior_vector_nrmse':nrm,'serialized_payload_bytes':len(payload),'bytes_per_behavior':len(payload)/BN,'private_fraction':1.0 if 'private_residual' in st else 0.0,'application_mac_proxy_per_behavior':mac,'decode_wall_seconds_bank':dt,'payload_sha256':digest,'max_replay_abs_error':rd,'payload_path':str(path.relative_to(ROOT.parents[2]))}

EVAL_TARGETS={}
def run(phase):
 global EVAL_TARGETS
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True);worlds=DEV if phase=='development' else FRESH;rows=[]
 for rho in [0.0,.1]:
  for world in worlds:
   for seed in SEEDS:
    cond,keys,coeff,vec,ang,truth,pred,held,probes=bank(world,seed,rho);EVAL_TARGETS[(world,seed,rho)]=vec.clone()
    for method in METHODS:rows.append(evalone(method,pack(method,cond,keys,coeff,vec),truth,pred,held,probes,phase,world,seed,rho))
 out=ART/('development.csv' if phase=='development' else 'fresh.csv')
 with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 (ROOT/'RESULTS_CORE.csv').write_text('')
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 (ART/('development_runs.jsonl' if phase=='development' else 'fresh_runs.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
 print(json.dumps({'phase':phase,'rows':len(rows)},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);a=p.parse_args();run(a.phase)
