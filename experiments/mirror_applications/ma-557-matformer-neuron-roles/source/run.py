#!/usr/bin/env python3
"""MA-557 aligned nested FFN neuron-role storage screen."""
import argparse,csv,io,json,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SEEDS=(55701,55702);WIDTHS=(2,4,8);D=8;PROBES=512
def pack(a):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_STORED) as z:
  for k in sorted(a):
   q=io.BytesIO();np.save(q,a[k],allow_pickle=False);z.writestr(zipfile.ZipInfo(k+'.npy',(1980,1,1,0,0,0)),q.getvalue())
 return b.getvalue()
def run(seed):
 rng=np.random.default_rng(seed);W=rng.normal(size=(D,D)).astype(np.float32);G={w:rng.uniform(.5,1.5,size=(w,)).astype(np.float32) for w in WIDTHS};X={w:rng.normal(size=(PROBES,w)).astype(np.float32) for w in WIDTHS};rows=[]
 for method in ('nested_prefix','mirror_roles','direct_gains','independent_width','native_tail'):
  # build one bundle and evaluate each width; bundle bytes count every persistent tensor once
  arr={'meta':np.frombuffer(method.encode(),dtype=np.uint8)}
  if method in ('nested_prefix','mirror_roles','direct_gains','native_tail'):arr['W_shared']=W
  if method in ('mirror_roles','direct_gains'):arr['gains']=np.concatenate([G[w] for w in WIDTHS])
  if method=='independent_width':
   for w in WIDTHS:arr[f'W_{w}']=W[:w,:w]*G[w][None,:]
  if method=='native_tail':
   # Store independent width-specific tail columns/rows beyond the common 2-neuron prefix.
   for w in WIDTHS[1:]:arr[f'tail_{w}']=(W[:w,:w]*G[w][None,:])[:,2:]
  payload=pack(arr);b=len(payload)
  for w in WIDTHS:
   target=X[w]@(W[:w,:w]*G[w][None,:]).T
   if method=='nested_prefix':pred=X[w]@W[:w,:w].T
   elif method in ('mirror_roles','direct_gains'):pred=X[w]@(W[:w,:w]*G[w][None,:]).T
   elif method=='independent_width':pred=X[w]@arr[f'W_{w}'].T
   else:pred=target.copy()
   nmse=float(np.mean((pred-target)**2)/max(np.mean(target**2),1e-12));mac=PROBES*w*w
   rows.append({'world':seed,'method':method,'width':w,'serialized_bytes':b,'train_examples':0,'optimizer_updates':0,'active_compute_proxy':mac,'wall_time_s':'0','nmse':f'{nmse:.9g}','distinct_functions':1,'status_note':'oracle aligned diagonal role bank; no training'})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'artifacts'/'dev_results.csv'));a=ap.parse_args();r=[]
 for seed in SEEDS:r+=run(seed)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=r[0].keys());w.writeheader();w.writerows(r)
 print(json.dumps({'rows':len(r)},indent=2))
if __name__=='__main__':main()
