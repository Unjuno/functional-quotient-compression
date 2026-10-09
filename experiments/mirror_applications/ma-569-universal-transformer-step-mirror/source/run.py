#!/usr/bin/env python3
import argparse,csv,io,json,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SEEDS=(56901,56902);D=8;STEPS=8;PROBES=512
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
 rng=np.random.default_rng(seed);A=rng.normal(size=(D,D))/np.sqrt(D);theta=np.array([.15+.06*t for t in range(STEPS)]);M=np.stack([rot(t)@A for t in theta]);x=rng.normal(size=(STEPS,PROBES,D));target=np.stack([x[t]@M[t].T for t in range(STEPS)]);rows=[]
 # Best rank-one residual factors are a strong oracle static-LoRA control.
 U=[];V=[]
 for t in range(STEPS):
  d=M[t]-A;u,s,v=np.linalg.svd(d,full_matrices=False);U.append((u[:,0]*s[0]).astype(np.float32));V.append(v[0].astype(np.float32))
 states={'hard_tied':{'A':A.astype(np.float32),'meta':np.frombuffer(b'tied',dtype=np.uint8)},
 'mirror_step':{'A':A.astype(np.float32),'angle_intercept':np.array([.15],np.float32),'angle_slope':np.array([.06],np.float32),'meta':np.frombuffer(b'mirror-step',dtype=np.uint8)},
 'direct_time_code':{'A':A.astype(np.float32),'angle_intercept':np.array([.15],np.float32),'angle_slope':np.array([.06],np.float32),'meta':np.frombuffer(b'direct-time',dtype=np.uint8)},
 'rank1_lora':{'U':np.stack(U),'V':np.stack(V),'meta':np.frombuffer(b'rank1-lora',dtype=np.uint8)},
 'untied':{'M':M.astype(np.float32),'meta':np.frombuffer(b'untied',dtype=np.uint8)}}
 for method,a in states.items():
  errs=[]
  for t in range(1,STEPS,2):
   if method=='hard_tied':pred=x[t]@A.T
   elif method in ('mirror_step','direct_time_code'):pred=x[t]@(rot(.15+.06*t)@A).T
   elif method=='rank1_lora':pred=x[t]@(A+np.outer(U[t],V[t])).T
   else:pred=x[t]@M[t].T
   errs.append(np.mean((pred-target[t])**2)/max(np.mean(target[t]**2),1e-12))
  mac=4*PROBES*D*D;tbytes=len(pack(a))
  rows.append({'world':seed,'method':method,'step':'odd-heldout','serialized_bytes':tbytes,'train_examples':4*PROBES,'optimizer_updates':0,'recurrent_mac_proxy':mac,'wall_time_s':'0','nmse':f'{np.mean(errs):.9g}','status_note':'teacher uses smooth linear Givens step angle; rank1 oracle control'})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'artifacts'/'dev_results.csv'));a=ap.parse_args();r=[]
 for s in SEEDS:r+=run(s)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=r[0].keys());w.writeheader();w.writerows(r)
 print(json.dumps({'rows':len(r)},indent=2))
if __name__=='__main__':main()
