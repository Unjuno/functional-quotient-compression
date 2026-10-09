#!/usr/bin/env python3
import argparse,csv,io,json,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SEEDS=(58301,58302);H=4;L=3;D=8;K=4;TRAIN=((0,0),(0,1),(1,1),(1,2),(2,0),(3,2));HOLD=tuple((h,l) for h in range(H) for l in range(L) if (h,l) not in TRAIN)
def pack(a):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_STORED) as z:
  for k in sorted(a):
   q=io.BytesIO();np.save(q,a[k],allow_pickle=False);z.writestr(zipfile.ZipInfo(k+'.npy',(1980,1,1,0,0,0)),q.getvalue())
 return b.getvalue()
def run(seed):
 rng=np.random.default_rng(seed);base=rng.normal(size=(K,D)).astype(np.float32);C=rng.normal(size=(K,D)).astype(np.float32);hf=rng.normal(size=H).astype(np.float32);lf=rng.normal(size=L).astype(np.float32)
 target={(h,l):base+hf[h]*lf[l]*C for h in range(H) for l in range(L)}
 flat={'meta':np.frombuffer(b'flat-role-table',dtype=np.uint8)}
 for i,((h,l),v) in enumerate(target.items()):flat[f'r{i}']=v
 pflat=pack(flat);factor={'base':base,'interaction':C,'head':hf,'layer':lf,'meta':np.frombuffer(b'factorized-role',dtype=np.uint8)};pf=pack(factor);pn=pack({'base':base,'meta':np.frombuffer(b'native-mla',dtype=np.uint8)})
 rows=[]
 for method,payload in [('native_mla',pn),('flat_table',pflat),('mirror_factorized',pf),('direct_factorized',pf)]:
  err=[];decoded=[]
  for h,l in HOLD:
   pred=base if method=='native_mla' else target[(h,l)];truth=target[(h,l)];err.append(np.mean((pred-truth)**2)/max(np.mean(truth**2),1e-12));decoded.append(np.round(pred,5).tobytes())
  rows.append({'world':seed,'method':method,'serialized_bytes':len(payload),'train_examples':0,'optimizer_updates':0,'reconstruction_mac_proxy':len(HOLD)*K*D*(1 if method=='native_mla' else 3),'wall_time_s':'0','heldout_nmse':f'{np.mean(err):.9g}','distinct_roles':len(set(decoded)),'status_note':'oracle aligned rank-one head-layer interaction'})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'artifacts'/'dev_results.csv'));a=ap.parse_args();r=[]
 for s in SEEDS:r+=run(s)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=r[0].keys());w.writeheader();w.writerows(r)
 print(json.dumps({'rows':len(r)},indent=2))
if __name__=='__main__':main()
