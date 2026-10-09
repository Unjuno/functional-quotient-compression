"""MA-327 factorized layer×expert Tucker coefficient generalization."""
import argparse,csv,hashlib,json,struct
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];D,K,L,E,T=8,8,16,16,32;DEV=[32700,32701];FRESH=[32710,32711,32712];SEEDS=[0,1,2]
torch.set_num_threads(2)
def rot(z,t):
 q=z.reshape(-1,2);c=torch.cos(t);s=torch.sin(t);return torch.stack([c*q[:,0]-s*q[:,1],s*q[:,0]+c*q[:,1]],-1).reshape(-1)
def make(w,seed,stratum):
 g=torch.Generator().manual_seed(w*100003+seed*7919+327);raw=torch.randn(D*D,K,generator=g);q=torch.linalg.qr(raw).Q.T;bank=q.reshape(K,D,D)
 layer=torch.arange(L).float()[:,None];freq=torch.arange(K).float()[None,:];A=1+0.15*torch.sin(.3*layer+.7*freq)
 if stratum=='aligned':z=torch.ones(K);theta=torch.linspace(0,2*torch.pi,E);theta[0]=0;B=torch.stack([rot(z,t) for t in theta])
 else:B=torch.randn(E,K,generator=g);B[0]=1;z=B[0].clone();theta=torch.zeros(E)
 coeff=A[:,None,:]*B[None,:,:];mats=torch.einsum('lek,kdh->ledh',coeff,bank);x=torch.randn(L,E,T,D,generator=g);y=torch.einsum('letd,ledh->leth',x,mats)
 obs=torch.ones(L,E,dtype=torch.bool)
 for l in range(L):
  for e in range(1,E):obs[l,e]=((l*17+e*13+seed)%4)!=0
 held=~obs
 return bank,A,B,coeff,mats,x,y,obs,held,z,theta
def pack(name,parts):
 m={'method':name,'shapes':[list(t.shape) for t in parts],'dtypes':[str(t.dtype) for t in parts],'layout':'layer×expert×Tucker-bank coefficient product'};b=json.dumps(m,sort_keys=True,separators=(',',':')).encode();return b'MA327\0'+struct.pack('<I',len(b))+b+b''.join(t.contiguous().numpy().tobytes() for t in parts)
def err(a,b,mask):
 x=a[mask];y=b[mask];return float((x-y).norm()/y.norm().clamp_min(1e-12))
def run(phase):
 rows=[]
 for w in DEV if phase=='development' else FRESH:
  for seed in SEEDS:
   for st in ('aligned','independent'):
    bank,A,B,C,mats,x,y,obs,held,ztrue,thetatrue=make(w,seed,st);ahat=C[:,0,:].clone();bhat=torch.zeros(E,K)
    for e in range(E):
     if e==0:bhat[e]=1
     else:
      ids=obs[:,e];bhat[e]=(C[ids,e,:]/ahat[ids]).mean(0)
    cfree=ahat[:,None,:]*bhat[None,:,:]
    # Mirror expert-side factor from observed ratios, with layer factor anchored at expert 0.
    z=torch.ones(K);jz=torch.zeros_like(z);v=z.reshape(-1,2);jz.reshape(-1,2)[:,0]=-v[:,1];jz.reshape(-1,2)[:,1]=v[:,0];angles=[]
    for e in range(E):
     if e==0:angles.append(torch.tensor(0.))
     else:
      ids=obs[:,e];r=(C[ids,e,:]/ahat[ids]).mean(0);aa=r.dot(z)/z.dot(z);bb=r.dot(jz)/jz.dot(jz);angles.append(torch.atan2(bb,aa))
    angles=torch.stack(angles);bmir=torch.stack([rot(z,t) for t in angles]);cmir=ahat[:,None,:]*bmir[None,:,:]
    cflat=C.clone();cflat[held]=0
    methods={'independent_full_matrices':(mats.half().float(),[mats.half()]),'flat_tucker_coefficients':(torch.einsum('lek,kdh->ledh',cflat.half().float(),bank.half().float()),[bank.half(),cflat.half()]),'factorized_free_layer_expert':(torch.einsum('lek,kdh->ledh',cfree.half().float(),bank.half().float()),[bank.half(),ahat.half(),bhat.half()]),'mirror_factorized_address':(torch.einsum('lek,kdh->ledh',cmir.half().float(),bank.half().float()),[bank.half(),ahat.half(),z.half(),angles.half()])}
    for name,(pred,parts) in methods.items():
     out=torch.einsum('letd,ledh->leth',x,pred);payload=pack(name,parts);rows.append({'phase':phase,'world':w,'seed':seed,'stratum':st,'method':name,'heldout_output_nrmse':err(out,y,held),'observed_output_nrmse':err(out,y,obs),'heldout_matrix_nrmse':err(pred,mats,held),'payload_bytes':len(payload),'observed_combinations':int(obs.sum()),'heldout_combinations':int(held.sum()),'payload_sha256':hashlib.sha256(payload).hexdigest()})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
