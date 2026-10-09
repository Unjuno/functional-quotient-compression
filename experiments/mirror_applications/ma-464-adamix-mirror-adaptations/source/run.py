#!/usr/bin/env python3
import csv,hashlib,io,json,math,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';DEV=[46400,46401];FRESH=[46410,46411,46412];SEEDS=[0,1,2];UPDATES=1000
def rot(a):
 c,s=torch.cos(a),torch.sin(a);return torch.stack([c,-s,s,c]).reshape(2,2)
def dataset(w,s):
 g=torch.Generator().manual_seed(w*1000003+s*997+53);W=torch.randn(2,2,generator=g)*.25+torch.eye(2)*.7;angles=torch.tensor([-1.2,-.4,.5,1.15]);D=torch.randn(2,2,generator=torch.Generator().manual_seed(w*31337+s*199))*.35
 mats=[rot(angles[i])@W for i in range(3)]+[W+D]
 xs=[];ys=[];qs=[];qys=[]
 for k,A in enumerate(mats):
  x=torch.randn(64,2,generator=torch.Generator().manual_seed(w*99131+s*71+k*23));q=torch.randn(256,2,generator=torch.Generator().manual_seed(w*1217+s*79+k*31));xs.append(x);ys.append(x@A.T);qs.append(q);qys.append(q@A.T)
 return mats,xs,ys,qs,qys

def train(w,s):
 torch.manual_seed(w*1009+s*101+7);mats,xs,ys,qs,qys=dataset(w,s)
 A=[torch.nn.Parameter(torch.zeros(2,2)) for _ in range(4)];opts=[torch.optim.Adam([a],lr=.035) for a in A]
 W=torch.nn.Parameter(torch.eye(2)*.5);ang=torch.nn.Parameter(torch.zeros(4));om=torch.optim.Adam([W,ang],lr=.025)
 t=time.perf_counter()
 for step in range(UPDATES):
  k=int(torch.randint(4,(1,)))
  ix=torch.randint(64,(16,));xb=xs[k][ix];yb=ys[k][ix]
  opts[k].zero_grad();loss=(xb@A[k].T-yb).square().mean();loss.backward();opts[k].step()
  om.zero_grad();pred=xb@(rot(ang[k])@W).T;loss=(pred-yb).square().mean();loss.backward();om.step()
 fitwall=time.perf_counter()-t
 with torch.no_grad():
  ada=torch.stack([a.detach() for a in A]);mir=torch.stack([rot(ang[k].detach())@W.detach() for k in range(4)])
  # pooled shared-only baseline, ordinary one matrix fit
  X=torch.cat(xs);Y=torch.cat(ys);shared=torch.linalg.solve(X.T@X+torch.eye(2)*1e-6,X.T@Y).T
 return mats,xs,ys,qs,qys,ada,mir,shared,fitwall,W.detach(),ang.detach()

def score(Ms,qs,qys,merged=False):
 errs=[];walls=[]
 for k in range(4):
  M=Ms if merged else Ms[k];t=time.perf_counter();o=qs[k]@M.T;walls.append(time.perf_counter()-t);errs.append(float((o-qys[k]).square().mean().sqrt()/(qys[k].square().mean().sqrt()+1e-9)))
 return statistics.mean(errs),statistics.mean(walls)
def package(method,state,N):
 obj={'format':'ma464-v1','method':method,'N':N}
 if method in ('adamix_routed','independent'):obj['adapters']=state[:N];obj['skill_route_ids']=torch.arange(N,dtype=torch.uint8)
 elif method=='mirror_routed':obj['shared_basis']=state[0];obj['view_angles']=state[1][:N];obj['skill_route_ids']=torch.arange(N,dtype=torch.uint8)
 elif method.endswith('_merged'):obj['merged_adapter']=state
 else:obj['shared_adapter']=state
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def run_phase(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True);worlds=DEV if phase=='development' else FRESH
 rows=[]
 for w in worlds:
  for s in SEEDS:
   mats,xs,ys,qs,qys,ada,mir,shared,wall,mirror_W,mirror_angles=train(w,s)
   metrics={'adamix_routed':score(ada,qs,qys),'independent':score(mats,qs,qys),'mirror_routed':score(mir,qs,qys),'adamix_merged':score(ada.mean(0),qs,qys,True),'mirror_merged':score(mir.mean(0),qs,qys,True),'shared':score(shared,qs,qys,True)}
   if phase=='development':rows.append({'world':w,'seed':s,'metrics':{k:v[0] for k,v in metrics.items()}});continue
   for method in metrics:
    state={'adamix_routed':ada,'independent':torch.stack(mats),'mirror_routed':(mirror_W,mirror_angles),'adamix_merged':ada.mean(0),'mirror_merged':mir.mean(0),'shared':shared}[method]
    # Keep original physical Mirror parameters in routed package.
    if method=='mirror_routed': state=(mirror_W,mirror_angles)
    for N in [1,4]:
     payload=package(method,state,N);path=PAY/f'{w}_{s}_{method}_N{N}.pt';path.write_bytes(payload)
     err,qwall=metrics[method]
     rows.append({'world':w,'seed':s,'method':method,'n':N,'skills':N,'nrmse_mean':err,'payload_bytes':len(payload),'bytes_per_skill':len(payload)/N,'optimizer_updates':UPDATES,'active_MAC_per_example':12 if method.startswith('mirror') else 8,'merge_MAC_one_time':16 if method.endswith('_merged') else 0,'fit_wall_seconds':wall,'query_wall_seconds_mean':qwall,'hash':hashlib.sha256(payload).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 if phase=='development':
  (ART/'development_runs.json').write_text(json.dumps(rows,indent=2)+'\n');return
 with open(ART/'fresh_runs.jsonl','w') as f:f.write(''.join(json.dumps(r)+'\n' for r in rows))
 with open(ROOT/'RESULTS_CORE.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['development','fresh'],required=True);run_phase(a.parse_args().phase)
