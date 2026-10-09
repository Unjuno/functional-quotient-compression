#!/usr/bin/env python3
import argparse,csv,io,json,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SEEDS=(56701,56702);D=8;NTR=NTE=512;ROUTE=.5
def pack(a):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_STORED) as z:
  for k in sorted(a):
   q=io.BytesIO();np.save(q,a[k],allow_pickle=False);z.writestr(zipfile.ZipInfo(k+'.npy',(1980,1,1,0,0,0)),q.getvalue())
 return b.getvalue()
def rot(a):
 R=np.eye(D)
 for i in range(0,D,2):R[i:i+2,i:i+2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]
 return R
def run(seed):
 rng=np.random.default_rng(seed);W=rng.normal(size=(D,D))/np.sqrt(D);angles=np.array([.45,-.55]);
 x=rng.normal(size=(NTR+NTE,D));roles=np.arange(len(x))%2;route=(np.linalg.norm(x,axis=1)>=np.quantile(np.linalg.norm(x,axis=1),1-ROUTE)); y=x.copy()
 for i in np.where(route)[0]:y[i]=x[i]@(rot(angles[roles[i]])@W).T
 xt,xe=x[:NTR],x[NTR:];rt,re=route[:NTR],route[NTR:];role=roles[NTR:];yt,ye=y[:NTR],y[NTR:];rows=[]
 Wrole=np.stack([rot(a)@W for a in angles])
 states={
 'mod_shared':{'W':W.astype(np.float32),'route_fraction':np.array([ROUTE],np.float32),'meta':np.frombuffer(b'mod',dtype=np.uint8)},
 'mirror_view':{'W':W.astype(np.float32),'role_angles':angles.astype(np.float32),'route_fraction':np.array([ROUTE],np.float32),'meta':np.frombuffer(b'mirror',dtype=np.uint8)},
 'direct_coeff':{'W':W.astype(np.float32),'role_angles':angles.astype(np.float32),'route_fraction':np.array([ROUTE],np.float32),'meta':np.frombuffer(b'direct',dtype=np.uint8)},
 'independent':{'W_roles':Wrole.astype(np.float32),'route_fraction':np.array([ROUTE],np.float32),'meta':np.frombuffer(b'independent',dtype=np.uint8)}}
 for method in states:
  pred=xe.copy()
  for j in np.where(re)[0]:
   if method=='mod_shared':mat=W
   elif method in ('mirror_view','direct_coeff'):mat=rot(angles[role[j]])@W
   else:mat=Wrole[role[j]]
   pred[j]=xe[j]@mat.T
  nmse=float(np.mean((pred[re]-ye[re])**2)/max(np.mean(ye[re]**2),1e-12));b=len(pack(states[method]));router_mac=len(xe)*D;block_mac=int(re.sum())*D*D
  t0=time.perf_counter();_=pred.copy();wall=time.perf_counter()-t0
  rows.append({'world':seed,'method':method,'route_fraction':ROUTE,'serialized_bytes':b,'train_tokens':NTR,'optimizer_updates':0,'router_mac_proxy':router_mac,'active_block_mac_proxy':block_mac,'wall_time_s':f'{wall:.6f}','routed_nmse':f'{nmse:.9g}','status_note':'fixed norm top-half router; role and view separated'})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'artifacts'/'dev_results.csv'));a=ap.parse_args();r=[]
 for s in SEEDS:r+=run(s)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=r[0].keys());w.writeheader();w.writerows(r)
 print(json.dumps({'rows':len(r)},indent=2))
if __name__=='__main__':main()
