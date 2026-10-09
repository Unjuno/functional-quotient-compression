#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads'
D,R,C,B=64,4,12,10;DEV=[51100,51101];FRESH=[51120,51121,51122];SEEDS=[0,1,2];STEPS=[400,800];METHODS=['pair_table','shared_tie','diag_factor','generic_full','mirror_givens']
torch.set_num_threads(1)
GG=torch.Generator().manual_seed(5110001);BASIS=torch.linalg.qr(torch.randn(D,R,generator=GG)).Q.contiguous()

def rotate(z,ang):
 out=z.unsqueeze(0).expand(C,-1,-1).clone()
 a,b=ang[:,0:1],ang[:,1:2]
 x,y=out[:,:,0].clone(),out[:,:,1].clone();ca,sa=a.cos(),a.sin();out[:,:,0]=ca*x-sa*y;out[:,:,1]=sa*x+ca*y
 x,y=out[:,:,2].clone(),out[:,:,3].clone();cb,sb=b.cos(),b.sin();out[:,:,2]=cb*x-sb*y;out[:,:,3]=sb*x+cb*y
 return out

def world_data(w,seed,rho):
 g=torch.Generator().manual_seed(w*100003+seed*7919+511)
 z=torch.randn(B,R,generator=g)*.5;ang=(torch.rand(C,2,generator=g)-.5)*2.0
 target=rotate(z,ang)
 if rho:target=target+rho*torch.randn(C,B,R,generator=g)
 held=torch.tensor([[(c*7+b*11+seed)%5==0 for b in range(B)] for c in range(C)])
 return target,held

def predict(method,state):
 if method=='pair_table':return state['coefficients'].float()
 z=state['behavior_codes']
 if method=='shared_tie':return z.unsqueeze(0).expand(C,-1,-1)
 if method=='diag_factor':return state['condition_scale'][:,None,:]*z[None,:,:]
 if method=='generic_full':return torch.einsum('crs,bs->cbr',state['condition_matrix'],z)
 if method=='mirror_givens':return rotate(z,state['condition_angles'])
 raise ValueError(method)

def fit(method,target,held,steps,seed):
 if method=='pair_table':return {'method':method,'basis':BASIS,'coefficients':target.clone()},0.0
 gen=torch.Generator().manual_seed(seed+METHODS.index(method)*77)
 z=torch.nn.Parameter(torch.randn(B,R,generator=gen)*.1);params=[z]
 state={'method':method,'basis':BASIS}
 if method=='mirror_givens':
  ang=torch.nn.Parameter(torch.zeros(C,2));params.append(ang);state['condition_angles']=ang
 elif method=='generic_full':
  mat=torch.nn.Parameter(torch.eye(R).unsqueeze(0).repeat(C,1,1));params.append(mat);state['condition_matrix']=mat
 elif method=='diag_factor':
  scale=torch.nn.Parameter(torch.ones(C,R));params.append(scale);state['condition_scale']=scale
 elif method=='shared_tie':pass
 opt=torch.optim.Adam(params,lr=.03);train=~held;t=time.perf_counter()
 for _ in range(steps):
  opt.zero_grad(set_to_none=True);pred=predict(method,state);loss=((pred[train]-target[train])**2).mean();loss.backward();opt.step()
 elapsed=time.perf_counter()-t
 result={'method':method,'basis':BASIS,'behavior_codes':z.detach().clone()}
 if method=='mirror_givens':result['condition_angles']=ang.detach().clone()
 elif method=='generic_full':result['condition_matrix']=mat.detach().clone()
 elif method=='diag_factor':result['condition_scale']=scale.detach().clone()
 return result,elapsed

def serialize(state):
 f=io.BytesIO();torch.save(state,f);return f.getvalue()

def evaluate(method,state,target,held,steps,fit_s,phase,w,seed,rho):
 payload=serialize(state);st=torch.load(io.BytesIO(payload),weights_only=False);pred=predict(method,st);hidden=held;obs=~held
 def nrm(mask):return float((pred[mask]-target[mask]).norm()/(target[mask].norm()+1e-12))
 for _ in range(3):
  if method=='pair_table':q=st['coefficients'].float()
  else:q=predict(method,st)
  _=q@st['basis'].T
 t=time.perf_counter()
 for _ in range(30):
  q=st['coefficients'].float() if method=='pair_table' else predict(method,st)
  _=q@st['basis'].T
 dt=(time.perf_counter()-t)/30
 mac={'pair_table':D*R,'shared_tie':D*R+R,'diag_factor':D*R+R,'generic_full':D*R+R*R,'mirror_givens':D*R+2*R}[method]
 path=PAY/phase/str(w)/str(seed)/f'rho{rho:.1f}'/(method+'.pt');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload);digest=hashlib.sha256(payload).hexdigest();re=torch.load(io.BytesIO(payload),weights_only=False);rp=predict(method,re);rd=float((rp-pred).abs().max())
 return {'phase':phase,'world':w,'seed':seed,'rho':rho,'steps':steps,'method':method,'heldout_nrmse':nrm(hidden),'observed_nrmse':nrm(obs),'serialized_payload_bytes':len(payload),'bytes_per_pair':len(payload)/(C*B),'optimizer_updates':0 if method=='pair_table' else steps,'fit_wall_seconds':fit_s,'decode_wall_seconds_bank':dt,'application_mac_proxy_per_pair':mac,'payload_sha256':digest,'max_replay_abs_error':rd,'payload_path':str(path.relative_to(ROOT.parents[2]))}

def run(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True);rows=[]
 if phase=='development':
  for steps in STEPS:
   for w in DEV:
    for seed in SEEDS:
     target,held=world_data(w,seed,0.0);st,sec=fit('mirror_givens',target,held,steps,w*100+seed);rows.append(evaluate('mirror_givens',st,target,held,steps,sec,phase,w,seed,0.0))
  with (ART/'development.csv').open('w',newline='') as f:q=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');q.writeheader();q.writerows(rows)
  scores={k:statistics.mean(float(x['heldout_nrmse']) for x in rows if x['steps']==k) for k in STEPS};selected=min(STEPS,key=lambda k:scores[k]);(ART/'selected_steps.json').write_text(json.dumps({'experiment_id':'MA-511','steps':selected,'development_mean_heldout_nrmse':scores,'fresh_worlds_locked':FRESH,'seeds':SEEDS},indent=2)+'\n')
 else:
  selected=json.loads((ART/'selected_steps.json').read_text())['steps']
  for rho in [0.0,.1]:
   for w in FRESH:
    for seed in SEEDS:
     target,held=world_data(w,seed,rho)
     for method in METHODS:
      st,sec=fit(method,target,held,selected,w*100+seed);rows.append(evaluate(method,st,target,held,selected,sec,phase,w,seed,rho))
  with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:q=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');q.writeheader();q.writerows(rows)
  (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
 print(json.dumps({'phase':phase,'rows':len(rows),'selected_steps':(selected if phase=='fresh' else json.loads((ART/'selected_steps.json').read_text())['steps'])},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);a=p.parse_args();run(a.phase)
