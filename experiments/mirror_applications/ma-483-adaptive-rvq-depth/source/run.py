#!/usr/bin/env python3
import csv,hashlib,io,json,math,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'; PAY=ART/'payloads'; DEV=[48300,48301]; FRESH=[48310,48311,48312]; SEEDS=[0,1,2]; D=16; N=128; K=32; THRESHOLDS=[0.,.01,.03,.06,.1]; AMPS=[1.,.5,.25,.125]

def make_world(w,s):
 g=torch.Generator().manual_seed(w*100003+s*7919+41); y=torch.zeros(N,D); active=torch.arange(N)%3; active=torch.where(active==0,1,torch.where(active==1,2,4)); active=active[torch.randperm(N,generator=g)]
 for j,a in enumerate(AMPS):
  ph=torch.rand(N,generator=g)*2*math.pi; mask=active>j; y[:,2*j]=mask*a*ph.cos();y[:,2*j+1]=mask*a*ph.sin()
 return active,y

def phase_code(y,j):
 ph=torch.atan2(y[:,2*j+1],y[:,2*j]); return torch.remainder(torch.round(ph*K/(2*math.pi)).long(),K)

def decode_codes(codes,depths):
 out=torch.zeros(len(depths),D); stage=[]
 for j in range(4):
  v=torch.zeros_like(out); mask=depths>j
  if mask.any():
   idx=codes[:,j].long();ph=idx*(2*math.pi/K);live=mask & (idx<K);v[live,2*j]=AMPS[j]*ph[live].cos();v[live,2*j+1]=AMPS[j]*ph[live].sin()
  out+=v;stage.append(v)
 return out,stage

def codes_for(y,method,tau=0):
 # The encoder sees y only. In this orthogonal task, an occupied plane has norm AMPS[j].
 depth=torch.zeros(len(y),dtype=torch.long);codes=torch.zeros(len(y),4,dtype=torch.uint8)
 for j in range(4):
  present=torch.linalg.vector_norm(y[:,2*j:2*j+2],dim=1)>tau
  if method=='adaptive':
   go=present & (depth==j); depth[go]=j+1
   ids=phase_code(y,j);codes[go,j]=ids[go].to(torch.uint8)
  else:
   depth[:]=4;ids=phase_code(y,j);present=torch.linalg.vector_norm(y[:,2*j:2*j+2],dim=1)>0;codes[:,j]=torch.where(present,ids,torch.full_like(ids,K)).to(torch.uint8)
 return codes,depth

def pack(method,y,tau):
 codes,depth=codes_for(y,method,tau)
 if method=='adaptive':codes=torch.cat([codes[i,:d] for i,d in enumerate(depth.tolist())])
 obj={'format':'ma483-v1','method':method,'K':K,'N':len(y),'codes':codes,'depth':depth,'basis':torch.eye(D)[:,:2],'scales':torch.tensor(AMPS)}
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()

def score(y,obj):
 codes=obj['codes']
 if codes.ndim==1:
  full=torch.zeros(len(y),4,dtype=torch.uint8);pos=0
  for i,d in enumerate(obj['depth'].tolist()):full[i,:d]=codes[pos:pos+d];pos+=d
  codes=full
 z,_=decode_codes(codes,obj['depth']);return float((z-y).norm()/(y.norm()+1e-12)),float(obj['depth'].float().mean())

def run(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True)
 if phase=='development':
  scores=[]
  for tau in THRESHOLDS:
   es=[];bs=[];ds=[]
   for w in DEV:
    for s in SEEDS:
     _,y=make_world(w,s);blob=pack('adaptive',y,tau);obj=torch.load(io.BytesIO(blob),weights_only=False);e,d=score(y,obj);es.append(e);bs.append(len(blob));ds.append(d)
   scores.append({'threshold':tau,'mean_error':statistics.mean(es),'mean_bytes':statistics.mean(bs),'mean_depth':statistics.mean(ds)})
  eligible=[x for x in scores if x['mean_error']<=.08]
  selected=min(eligible,key=lambda x:x['mean_bytes']) if eligible else min(scores,key=lambda x:x['mean_error'])
  (ART/'development_selection.json').write_text(json.dumps({'sweep':scores,'selected_tau':selected['threshold']},indent=2)+'\n');print(json.dumps(selected));return
 sel=json.loads((ART/'development_selection.json').read_text());tau=sel['selected_tau'];rows=[]
 for w in FRESH:
  for s in SEEDS:
   _,y=make_world(w,s)
   for method in ['fixed','adaptive']:
    for threshold in ([tau] if method=='adaptive' else [0.]):
     blob=pack(method,y,threshold);obj=torch.load(io.BytesIO(blob),weights_only=False);err,depth=score(y,obj)
     path=PAY/f'{w}_{s}_{method}.pt';path.write_bytes(blob); rows.append({'world':w,'seed':s,'method':method,'threshold':threshold,'normalized_rmse':err,'mean_active_stages':depth,'payload_bytes':len(blob),'bytes_per_function':len(blob)/N,'decode_MAC_per_function':depth*4,'hash':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));print(json.dumps({'phase':phase,'rows':len(rows),'tau':tau}))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
