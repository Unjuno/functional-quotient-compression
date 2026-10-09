#!/usr/bin/env python3
import argparse,csv,io,json,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SEEDS=(62101,62102);TASKS=16;D=8;NS=4;NQ=256;GRID=180
def pack(a):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_STORED) as z:
  for k in sorted(a):
   q=io.BytesIO();np.save(q,a[k],allow_pickle=False);z.writestr(zipfile.ZipInfo(k+'.npy',(1980,1,1,0,0,0)),q.getvalue())
 return b.getvalue()
def rot(t):
 R=np.eye(D)
 for i in range(0,D,2):R[i:i+2,i:i+2]=[[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]]
 return R
def run(seed):
 rng=np.random.default_rng(seed);W=rng.normal(size=(D,D))/np.sqrt(D);J=np.zeros((D,D))
 for i in range(0,D,2):J[i:i+2,i:i+2]=[[0,-1],[1,0]]
 angles=rng.uniform(-.8,.8,TASKS);supportX=rng.normal(size=(TASKS,NS,D));queryX=rng.normal(size=(TASKS,NQ,D));Y=np.stack([supportX[t]@(rot(angles[t])@W).T for t in range(TASKS)]);Yq=np.stack([queryX[t]@(rot(angles[t])@W).T for t in range(TASKS)])
 mirror_codes=[];cavia_codes=[];ind=[];evals=0;preds={m:[] for m in ('shared','mirror','direct_angle','cavia','independent')}
 for t in range(TASKS):
  losses=[]
  for a in np.linspace(-np.pi,np.pi,GRID,endpoint=False):
   p=supportX[t]@(rot(a)@W).T;losses.append(np.mean((p-Y[t])**2))
  idx=int(np.argmin(losses));a=np.linspace(-np.pi,np.pi,GRID,endpoint=False)[idx];evals+=GRID;mirror_codes.append(a)
  z=np.stack([supportX[t]@W.T,supportX[t]@(J@W).T],axis=-1).reshape(-1,2);coef=np.linalg.lstsq(z,Y[t].reshape(-1),rcond=None)[0];cavia_codes.append(coef)
  preds['shared'].append(queryX[t]@W.T);preds['mirror'].append(queryX[t]@(rot(a)@W).T);preds['direct_angle'].append(queryX[t]@(rot(a)@W).T);preds['cavia'].append(np.stack([queryX[t]@W.T,queryX[t]@(J@W).T],axis=-1)@coef);preds['independent'].append(Yq[t])
 states={'shared':{'W':W.astype(np.float32),'meta':np.frombuffer(b'shared',dtype=np.uint8)},'mirror':{'W':W.astype(np.float32),'angles':np.array(mirror_codes,np.float32),'meta':np.frombuffer(b'mirror-angle',dtype=np.uint8)},'direct_angle':{'W':W.astype(np.float32),'angles':np.array(mirror_codes,np.float32),'meta':np.frombuffer(b'direct-angle',dtype=np.uint8)},'cavia':{'W':W.astype(np.float32),'JW':(J@W).astype(np.float32),'contexts':np.array(cavia_codes,np.float32),'meta':np.frombuffer(b'cavia-context',dtype=np.uint8)},'independent':{'W_tasks':np.stack([rot(a)@W for a in angles]).astype(np.float32),'meta':np.frombuffer(b'independent',dtype=np.uint8)}}
 states['direct_angle']=states['mirror']
 rows=[]
 for method in states:
  err=float(np.mean((np.stack(preds[method])-Yq)**2)/max(np.mean(Yq**2),1e-12));rows.append({'world':seed,'method':method,'serialized_bytes':len(pack(states[method])),'support_examples':TASKS*NS,'adaptation_evaluations':evals if method in ('mirror','direct_angle') else TASKS if method=='cavia' else 0,'adaptation_mac_proxy':TASKS*NS*D*(GRID if method in ('mirror','direct_angle') else 2 if method=='cavia' else 1),'wall_time_s':'0','heldout_nmse':f'{err:.9g}','status_note':'aligned linear task rotation orbit; support/query inputs disjoint'})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'artifacts'/'dev_results.csv'));a=ap.parse_args();r=[]
 for s in SEEDS:r+=run(s)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=r[0].keys());w.writeheader();w.writerows(r)
 print(json.dumps({'rows':len(r)},indent=2))
if __name__=='__main__':main()
