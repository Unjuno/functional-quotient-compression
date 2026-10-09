"""MA-314 adaptive intrinsic dimensions with Mirror address and private fallback."""
import argparse,csv,hashlib,json,struct,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];D,d,N=128,32,24;RANKS=(4,8,16,32);DEV=[31400,31401];FRESH=[31410,31411,31412];SEEDS=[0,1,2];TOL=.05
torch.set_num_threads(2)
def rotate(z,t):
 q=z.reshape(-1,2);c=torch.cos(t);s=torch.sin(t);return torch.stack([c*q[:,0]-s*q[:,1],s*q[:,0]+c*q[:,1]],-1).reshape(-1)
def quarter(z,k):return torch.cat([z[:k],torch.zeros_like(z[k:])])
def make(w,seed,stratum):
 g=torch.Generator().manual_seed(w*100003+seed*7919+314);U=torch.linalg.qr(torch.randn(D,d,generator=g)).Q[:,:d];ranks=torch.tensor([4,8,16,32]*6);htr=torch.randn(N,96,d,generator=g);hq=torch.randn(N,128,d,generator=g)
 if stratum=='aligned':
  z=torch.randn(d,generator=g);theta=torch.rand(N,generator=g)*2*torch.pi;theta[3]=0;coef=torch.stack([rotate(quarter(z,int(ranks[i])),theta[i]) for i in range(N)])
 else:coef=torch.randn(N,d,generator=g);z=coef[3].clone();theta=torch.zeros(N)
 ytr=torch.einsum('ntd,nd->nt',htr,coef);yq=torch.einsum('ntd,nd->nt',hq,coef);return U,ranks,htr,ytr,hq,yq,coef,z,theta
def pack(name,parts,meta):
 m={'method':name,'shapes':[list(t.shape) for t in parts],'dtypes':[str(t.dtype) for t in parts]};m.update(meta);b=json.dumps(m,sort_keys=True,separators=(',',':')).encode();return b'MA314\0'+struct.pack('<I',len(b))+b+b''.join(t.contiguous().numpy().tobytes() for t in parts)
def nrmse(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-12))
def run(phase):
 rows=[]
 for w in DEV if phase=='development' else FRESH:
  for seed in SEEDS:
   for st in ('aligned','independent'):
    U,true_r,htr,ytr,hq,yq,target,ztrue,theta_true=make(w,seed,st);t0=time.perf_counter();est=torch.stack([torch.linalg.lstsq(htr[i],ytr[i]).solution for i in range(N)]);direct_s=time.perf_counter()-t0;t0=time.perf_counter()
    # Per-task adaptive intrinsic dimension from cumulative coordinate residual.
    dims=[]
    for c in est:
     k=32
     for cand in RANKS:
      if float(c[cand:].norm()/c.norm().clamp_min(1e-12))<=.01:k=cand;break
     dims.append(k)
    dims=torch.tensor(dims);direct=torch.stack([quarter(est[i],int(dims[i])) for i in range(N)])
    # Estimate one shared vector from known full-rank anchor task 3 and infer task angles.
    z=est[3].clone();mirror=[];angles=[]
    for i in range(N):
     zk=quarter(z,int(dims[i]));jz=torch.zeros_like(zk);q=zk.reshape(-1,2);jz.reshape(-1,2)[:,0]=-q[:,1];jz.reshape(-1,2)[:,1]=q[:,0];den=zk.dot(zk).clamp_min(1e-9);a=est[i].dot(zk)/den;b=est[i].dot(jz)/den;t=torch.atan2(b,a);angles.append(t);mirror.append(rotate(zk,t))
    mirror=torch.stack(mirror);angles=torch.stack(angles);res=est-mirror;rel=res.norm(dim=1)/est.norm(dim=1).clamp_min(1e-12);split=rel>TOL;private=res.clone();private[~split]=0;private=torch.where(private.abs()>1e-6,private,torch.zeros_like(private));mirpriv=mirror+private;pi=torch.nonzero(private!=0,as_tuple=False).to(torch.int16);pv=private[private!=0].half();mirror_s=time.perf_counter()-t0;t0=time.perf_counter()
    # Generic PCA controls: choose smallest rank meeting 5% coefficient reconstruction tolerance.
    mean=est.mean(0);u,s,vh=torch.linalg.svd(est-mean,full_matrices=False);pr=d
    for r in RANKS:
     candidate=(est-mean)@vh[:r].T@vh[:r]+mean
     if nrmse(candidate,est)<=TOL:pr=r;break
    basis=vh[:pr];pc=(est-mean)@basis.T;pca=pc@basis+mean;pca_s=time.perf_counter()-t0
    methods={'direct_adaptive_intrinsic':(direct,[U.half(),dims.to(torch.uint8),torch.cat([est[i,:int(dims[i])] for i in range(N)]).half()],0,direct_s),'mirror_adaptive_view':(mirpriv,[U.half(),z.half(),angles.half(),dims.to(torch.uint8),pi,pv],int(split.sum()),mirror_s),'mirror_view_no_private':(mirror,[U.half(),z.half(),angles.half(),dims.to(torch.uint8)],0,mirror_s),'generic_pca_adaptive_rank':(pca,[U.half(),mean.half(),basis.half(),pc.half()],0,pca_s)}
    for name,(coef,parts,splitcnt,secs) in methods.items():
     out=torch.einsum('ntd,nd->nt',hq,coef);payload=pack(name,parts,{'D':D,'d':d,'N':N,'split_tasks':splitcnt});rows.append({'phase':phase,'world':w,'seed':seed,'stratum':st,'method':name,'query_nrmse':nrmse(out,yq),'payload_bytes':len(payload),'task_code_bytes':len(pack(name+'_task',parts[1:],{'N':N})),'active_dimension_mean':float(dims.float().mean()),'private_split_tasks':splitcnt,'private_values':int((private!=0).sum()) if name=='mirror_adaptive_view' else 0,'fit_seconds':secs,'support_examples':N*96,'optimizer_updates':0,'payload_sha256':hashlib.sha256(payload).hexdigest()})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
