#!/usr/bin/env python3
"""MA-437 synthetic S4-structured transition Mirror screen."""
import argparse, hashlib, json, math, random, time
from pathlib import Path
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts'; PAY=OUT/'payloads'
DEV=[43700,43701]; FRESH=[43710,43711,43712]; SEEDS=[0,1,2]
METHODS=['shared','gate','mirror','rank1','independent']; LRS=[0.003,0.01]
D=4; TRAIN_LEN=32; EVAL_LENS=[32,64]; UPDATES=500; BATCH=96; ROLES=4

def seedall(s): random.seed(s); torch.manual_seed(s); torch.set_num_threads(1)
def rot(a):
 c,s=a.cos(),a.sin(); return torch.stack([torch.stack([c,-s]),torch.stack([s,c])])
def embed_rot(a):
 q=torch.eye(D); q[:2,:2]=rot(a); return q

def world_params(w):
 g=torch.Generator().manual_seed(w)
 diag=-torch.rand(D,generator=g)*.18-.45
 u=torch.randn(D,generator=g); u=u/u.norm()*0.08
 v=torch.randn(D,generator=g); v=v/v.norm()*0.08
 angles=torch.tensor([-0.75,-0.25,0.35,0.8]) + (torch.rand(ROLES,generator=g)-.5)*.12
 return diag,u,v,angles

def stable_matrix(diag,u,v):
 # S4-style stable normal diagonal plus low-rank correction; spectral radius remains < 1.
 A=torch.diag(diag)+torch.outer(u,v)
 eig=torch.linalg.eigvals(A).abs().max().item()
 if eig>=0.9: A=A*(0.89/eig)
 return A

def teacher_mats(params):
 diag,u,v,angles=params; base=stable_matrix(diag,u,v)
 return torch.stack([stable_matrix(diag,embed_rot(a)@u,embed_rot(a)@v) for a in angles])

def make_data(w,n):
 params=world_params(w); mats=teacher_mats(params); g=torch.Generator().manual_seed(w+918273)
 x=torch.randn(n,D,generator=g)*.4; role=torch.randint(0,ROLES,(n,),generator=g); ys=[]
 for L in [TRAIN_LEN]:
  z=x.clone(); traj=[]
  # Fixed per-sequence role to isolate structured dynamics from routing.
  a=mats[role]
  for _ in range(L): z=torch.einsum('bij,bj->bi',a,z); traj.append(z)
  ys.append(torch.stack(traj,1))
 return x,role,ys[0],params

class Structured(nn.Module):
 def __init__(self,method,init):
  super().__init__(); self.method=method
  diag,u,v,angles=init
  if method=='independent':
   self.diag=nn.Parameter(diag.repeat(ROLES,1).clone()); self.u=nn.Parameter(u.repeat(ROLES,1).clone()); self.v=nn.Parameter(v.repeat(ROLES,1).clone())
  else:
   self.diag=nn.Parameter(diag.clone()); self.u=nn.Parameter(u.clone()); self.v=nn.Parameter(v.clone())
   if method=='gate': self.g=nn.Parameter(torch.zeros(ROLES,D))
   if method=='mirror': self.angle=nn.Parameter(torch.zeros(ROLES))
   if method=='rank1': self.du=nn.Parameter(torch.zeros(ROLES,D)); self.dv=nn.Parameter(torch.zeros(ROLES,D))
 def matrix(self,r):
  if self.method=='independent': return stable_matrix(self.diag[r],self.u[r],self.v[r])
  d,u,v=self.diag,self.u,self.v
  if self.method=='gate': d=d+self.g[r]*.08
  if self.method=='mirror':
   q=embed_rot(self.angle[r]); u=q@u; v=q@v
  if self.method=='rank1': u=u+self.du[r]; v=v+self.dv[r]
  return stable_matrix(d,u,v)
 def forward(self,x,role,length):
  z=x; out=[]; mats=torch.stack([self.matrix(r) for r in range(ROLES)])
  for _ in range(length):
   z=torch.bmm(mats[role],z.unsqueeze(-1)).squeeze(-1); out.append(z)
  return torch.stack(out,1)

def nrmse(pred,target): return ((pred-target).square().mean().sqrt()/(target.square().mean().sqrt()+1e-12)).item()
def serial(model,method,world,seed,phase):
 payload={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
 data={'method':method,'world':world,'seed':seed,'phase':phase,'state':payload,'format':'torch-save-v1'}
 path=PAY/f'{phase}_{world}_{seed}_{method}.pt'; torch.save(data,path)
 b=path.read_bytes(); return len(b),hashlib.sha256(b).hexdigest(),path

def replay_payload(path):
 data=torch.load(path,map_location='cpu',weights_only=False); model=Structured(data['method'],world_params(data['world'])); model.load_state_dict(data['state']); return model

def fit(world,seed,method,lr,phase):
 seedall(world*100+seed*7+sum(map(ord,method)))
 init=world_params(world); x,role,y,_=make_data(world,1024)
 model=Structured(method,init); opt=torch.optim.AdamW(model.parameters(),lr=lr)
 st=time.perf_counter()
 for step in range(UPDATES):
  idx=torch.randint(len(x),(BATCH,)); pred=model(x[idx],role[idx],TRAIN_LEN); loss=(pred-y[idx]).square().mean()
  opt.zero_grad(); loss.backward(); opt.step()
 train_wall=time.perf_counter()-st
 nbytes,digest,path=serial(model,method,world,seed,phase)
 return model,{'serialized_bytes':nbytes,'payload_sha256':digest,'payload_path':str(path.relative_to(ROOT.parents[2])),'train_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'train_wall_s':train_wall}

def evaluate(model,w,length):
 x,role,y,_=make_data(w,256); st=time.perf_counter()
 with torch.no_grad(): pred=model(x,role,length)
 wall=time.perf_counter()-st
 target=y if length==TRAIN_LEN else None
 if target is None:
  z=x.clone(); mats=teacher_mats(world_params(w)); ys=[]
  for _ in range(length): z=torch.einsum('bij,bj->bi',mats[role],z); ys.append(z)
  target=torch.stack(ys,1)
 stable=max(torch.linalg.eigvals(model.matrix(r)).abs().max().item() for r in range(ROLES))
 return nrmse(pred,target),wall,stable

def mac_proxy(method):
 # dense state application plus role-specific parameter generation cost; explicitly a proxy.
 base=2*D*D
 extra={'shared':0,'gate':D,'mirror':8,'rank1':2*D,'independent':0}[method]
 return base+extra

def run():
 ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['development','fresh'],required=True); a=ap.parse_args()
 OUT.mkdir(exist_ok=True); PAY.mkdir(parents=True,exist_ok=True); worlds=DEV if a.phase=='development' else FRESH
 rows=[]; selections={}
 if a.phase=='fresh':
  locked=json.loads((OUT/'development_selection.json').read_text())
  chosen=locked['selected_learning_rate_by_method']
 for w in worlds:
  for method in METHODS:
   if a.phase=='development':
    scores={lr:[] for lr in LRS}
    for lr in LRS:
     for seed in SEEDS:
      model,meta=fit(w,seed,method,lr,a.phase)
      val,wall,stable=evaluate(model,w,TRAIN_LEN); scores[lr].append(val)
      rows.append({'condition':'development','world_or_seed':f'{w}-{seed}','method':method,'serialized_bytes':meta['serialized_bytes'],'train_tokens_or_examples':meta['train_examples'],'optimizer_updates':UPDATES,'active_compute_proxy':f'{mac_proxy(method)} MAC/step','wall_time_s':round(meta['train_wall_s'],6),'primary_metric':'trajectory_NRMSE','primary_value':val,'secondary_metric':'spectral_radius','secondary_value':stable,'status_note':f'lr={lr}; val_wall={wall:.6f}; sha256={meta["payload_sha256"]}; payload={meta["payload_path"]}'})
    selections[method]=min(LRS,key=lambda lr:sum(scores[lr])/len(scores[lr]))
   else:
    lr=chosen[method]
    for seed in SEEDS:
     model,meta=fit(w,seed,method,lr,a.phase)
     for length in EVAL_LENS:
      metric,wall,stable=evaluate(model,w,length)
      rows.append({'condition':'fresh','world_or_seed':f'{w}-{seed}-{length}','method':method,'serialized_bytes':meta['serialized_bytes'],'train_tokens_or_examples':meta['train_examples'],'optimizer_updates':UPDATES,'active_compute_proxy':f'{mac_proxy(method)} MAC/step','wall_time_s':round(wall,6),'primary_metric':'trajectory_NRMSE','primary_value':metric,'secondary_metric':'spectral_radius','secondary_value':stable,'status_note':f'lr={lr}; train_wall={meta["train_wall_s"]:.6f}; sha256={meta["payload_sha256"]}; payload={meta["payload_path"]}'})
 if a.phase=='development':
  (OUT/'development_selection.json').write_text(json.dumps({'selected_learning_rate_by_method':selections,'rule':'lowest mean length-32 validation NRMSE across both development worlds and seeds 0,1,2','fresh_worlds_not_accessed':True},indent=2)+'\n')
 (OUT/f'{a.phase}_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 print(json.dumps({'phase':a.phase,'selection':selections if a.phase=='development' else chosen,'records':len(rows)},indent=2))

if __name__=='__main__': run()
