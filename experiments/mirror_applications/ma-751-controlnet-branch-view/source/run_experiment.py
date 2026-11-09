#!/usr/bin/env python3
"""Synthetic ControlNet-like spatial branch sharing screen."""
from __future__ import annotations
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
SEED=75101;C=8;HID=8;OUT=4;S=16;PLANES=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
torch.set_num_threads(1)

def pack(**a):
 b=io.BytesIO();np.savez_compressed(b,**a);return b.getvalue()
def rotation(angles):
 r=np.eye(HID,dtype=np.float32)
 for a,(i,j) in zip(angles,PLANES):
  c=np.cos(a);s=np.sin(a);g=np.eye(HID,dtype=np.float32);g[i,i]=g[j,j]=c;g[i,j]=-s;g[j,i]=s;r=g@r
 return r

def run():
 rng=np.random.default_rng(SEED)
 w1=rng.normal(0,.3,size=(HID,1,3,3)).astype(np.float32);w2=rng.normal(0,.24,size=(OUT,HID,1,1)).astype(np.float32)
 angles=rng.uniform(-.55,.55,size=(C,len(PLANES))).astype(np.float32)
 xtrain=torch.tensor(rng.normal(size=(C,32,1,S,S)).astype(np.float32));xtest=torch.tensor(rng.normal(size=(C,16,1,S,S)).astype(np.float32))
 tw1=torch.tensor(w1);tw2=torch.tensor(w2)
 def hidden(x):return torch.relu(F.conv2d(x,tw1,padding=1))
 target_train=[];target_test=[]
 for c in range(C):
  r=torch.tensor(rotation(angles[c]));target_train.append(F.conv2d(torch.einsum('ij,bjhw->bihw',r,hidden(xtrain[c])),tw2));target_test.append(F.conv2d(torch.einsum('ij,bjhw->bihw',r,hidden(xtest[c])),tw2))
 target_train=torch.stack(target_train);target_test=torch.stack(target_test)
 # Fit the simple per-channel output FiLM control on train maps only, matching rotated hidden states.
 gam=np.ones((C,HID),np.float32);bet=np.zeros_like(gam);htr=hidden(xtrain.reshape(C*32,1,S,S)).reshape(C,32,HID,S,S).detach().numpy()
 for c in range(C):
  hr=np.einsum('ij,bjhw->bihw',rotation(angles[c]),htr[c])
  for j in range(HID):
   q=htr[c,:,j].reshape(-1);y=hr[:,j].reshape(-1);a,b=np.polyfit(q,y,1);gam[c,j]=a;bet[c,j]=b
 wa=np.broadcast_to(w1,(C,)+w1.shape).copy();w2eff=np.stack([np.einsum('oi,ij->oj',w2[:,:,0,0],rotation(a))[:,:,None,None] for a in angles])
 gid=np.arange(C,dtype=np.int8);full=pack(copied_encoder=wa,copied_zero_convs=w2eff,condition_ids=gid)
 shared=pack(shared_encoder=w1,shared_zero_conv=w2,condition_ids=gid)
 film=pack(shared_encoder=w1,shared_zero_conv=w2,gamma=gam,beta=bet,condition_ids=gid)
 view=pack(shared_encoder=w1,shared_zero_conv=w2,givens_angles=angles,condition_ids=gid,plane_ids=np.asarray(PLANES,np.int8))
 methods=[('independent_copied',full,'full'),('hard_tied',shared,'hard'),('output_film',film,'film'),('native_direct_givens',view,'view'),('mirror_givens_view',view,'view')]
 conv_macs=(HID*1*9+OUT*HID)*S*S
 rows=[]
 for name,blob,kind in methods:
  errs=[];times=[]
  for c in range(C):
   xx=xtest[c];start=time.perf_counter()
   h=hidden(xx)
   if kind=='full':
    pred=F.conv2d(h,torch.tensor(w2eff[c]),padding=0)
   elif kind=='hard':pred=F.conv2d(h,tw2)
   elif kind=='film':pred=F.conv2d(h*torch.tensor(gam[c])[None,:,None,None]+torch.tensor(bet[c])[None,:,None,None],tw2)
   else:
    rr=torch.tensor(rotation(angles[c]));pred=F.conv2d(torch.einsum('ij,bjhw->bihw',rr,h),tw2)
   times.append((time.perf_counter()-start)*1000);errs.append(float(torch.linalg.vector_norm(pred-target_test[c])/torch.linalg.vector_norm(target_test[c])))
  extra=0 if kind in ('full','hard') else (len(PLANES)*4*S*S*C*16 if kind=='view' else HID*S*S*C*2)
  macs=C*16*conv_macs+extra
  rows.append(dict(seed=SEED,method=name,split='heldout_maps',serialized_bytes=len(blob),payload_sha256=hashlib.sha256(blob).hexdigest(),spatial_nrmse=float(np.mean(errs)),active_macs=macs,wall_ms=float(sum(times)),status_note='exact native Givens alias' if name=='mirror_givens_view' else ''))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('wrote',len(rows),'rows')
if __name__=='__main__':run()
