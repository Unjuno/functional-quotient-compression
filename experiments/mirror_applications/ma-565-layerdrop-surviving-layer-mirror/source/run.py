#!/usr/bin/env python3
import argparse,csv,io,json,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SEEDS=(56501,56502);D=8;DEPTHS=(2,3,4);NTR=NTE=512
def pack(a):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_STORED) as z:
  for k in sorted(a):
   q=io.BytesIO();np.save(q,a[k],allow_pickle=False);z.writestr(zipfile.ZipInfo(k+'.npy',(1980,1,1,0,0,0)),q.getvalue())
 return b.getvalue()
def run(seed):
 rng=np.random.default_rng(seed);blocks=rng.normal(0,.12,(4,D,D));x=rng.normal(size=(NTR+NTE,D));
 # Residual feature sum yields a linear shared full-depth target.
 full=np.eye(D)+blocks.sum(0);y=x@full.T;xtr,xte=x[:NTR],x[NTR:];yte=y[NTR:];rows=[]
 for depth in DEPTHS:
  active=np.arange(4) if depth==4 else np.linspace(0,3,depth,dtype=int)
  feats=np.stack([xtr@blocks[i].T for i in active],axis=-1) # N,D,K
  gates=np.linalg.lstsq(feats.reshape(-1,depth), (y[:NTR]-xtr).reshape(-1),rcond=None)[0]
  for method in ('layerdrop','mirror_role','direct_gate','independent'):
   if method=='layerdrop':pred=xte+sum(xte@blocks[i].T for i in active);arr={'blocks':blocks.astype(np.float32),'mask':active.astype(np.uint8),'meta':np.frombuffer(b'layerdrop',dtype=np.uint8)}
   elif method in ('mirror_role','direct_gate'):
    pred=xte+sum(gates[j]*(xte@blocks[i].T) for j,i in enumerate(active));arr={'blocks':blocks.astype(np.float32),'mask':active.astype(np.uint8),'gates':gates.astype(np.float32),'meta':np.frombuffer(b'role-code',dtype=np.uint8)}
   else:pred=xte@full.T;arr={'W_independent':full.astype(np.float32),'meta':np.frombuffer(b'independent',dtype=np.uint8)}
   nmse=float(np.mean((pred-yte)**2)/max(np.mean(yte**2),1e-12));mac=NTE*D*D*(depth if method!='independent' else 1)
   rows.append({'world':seed,'depth':depth,'method':method,'serialized_bytes':len(pack(arr)),'train_examples':NTR,'optimizer_updates':0,'active_compute_proxy':mac,'wall_time_s':'0','nmse':f'{nmse:.9g}','status_note':'oracle linear residual stack, gates fit by least squares'})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'artifacts'/'dev_results.csv'));a=ap.parse_args();r=[]
 for s in SEEDS:r+=run(s)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=r[0].keys());w.writeheader();w.writerows(r)
 print(json.dumps({'rows':len(r)},indent=2))
if __name__=='__main__':main()
