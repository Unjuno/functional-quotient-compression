"""MA-319 Tucker matrix bank with structured Mirror layer coefficients."""
import argparse,csv,hashlib,json,struct,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];D,K,L=32,8,64;DEV=[31900,31901];FRESH=[31910,31911,31912];SEEDS=[0,1,2]
torch.set_num_threads(2)
def rotate(z,t):
 q=z.reshape(-1,2);c=torch.cos(t);s=torch.sin(t);return torch.stack([c*q[:,0]-s*q[:,1],s*q[:,0]+c*q[:,1]],-1).reshape(-1)
def make(w,seed,stratum):
 g=torch.Generator().manual_seed(w*100003+seed*7919+319);bank=torch.randn(K,D,D,generator=g);bank=bank/bank.flatten(1).norm(dim=1)[:,None,None]
 if stratum=='aligned':
  z=torch.randn(K,generator=g);theta=torch.linspace(0,2*torch.pi,L);theta[0]=0;coef=torch.stack([rotate(z,t) for t in theta])
 else:coef=torch.randn(L,K,generator=g);z=coef[0].clone();theta=torch.zeros(L)
 mats=torch.einsum('lk,kij->lij',coef,bank);x=torch.randn(L,128,D,generator=g);y=torch.einsum('lbd,ldf->lbf',x,mats);return bank,coef,z,theta,x,y,mats
def pack(name,parts,meta):
 md={'method':name,'shapes':[list(t.shape) for t in parts],'dtypes':[str(t.dtype) for t in parts]};md.update(meta);b=json.dumps(md,sort_keys=True,separators=(',',':')).encode();return b'MA319\0'+struct.pack('<I',len(b))+b+b''.join(t.contiguous().numpy().tobytes() for t in parts)
def err(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-12))
def run(phase):
 rows=[]
 for w in DEV if phase=='development' else FRESH:
  for seed in SEEDS:
   for st in ('aligned','independent'):
    bank,c,ztrue,theta,x,y,mats=make(w,seed,st)
    # Mandatory Tucker baseline: free layer coefficient vector over shared matrix bank.
    free=torch.einsum('lbd,ldf->lbf',x,mats)
    t0=time.perf_counter();z=c[0].clone();Jz=torch.zeros_like(z);Jz[0::2]=-z[1::2];Jz[1::2]=z[0::2];angles=[]
    for row in c:
     a=row.dot(z)/z.dot(z).clamp_min(1e-9);b=row.dot(Jz)/Jz.dot(Jz).clamp_min(1e-9);angles.append(torch.atan2(b,a))
    angles=torch.stack(angles);mc=torch.stack([rotate(z,t) for t in angles]);mirror=torch.einsum('lk,kij->lij',mc,bank);tm=time.perf_counter()-t0
    t0=time.perf_counter();mean=c.mean(0);u,s,vh=torch.linalg.svd(c-mean,full_matrices=False);basis=vh[:2];pc=(c-mean)@basis.T;pca_c=pc@basis+mean;pca=torch.einsum('lk,kij->lij',pca_c,bank);tp=time.perf_counter()-t0
    bi=bank.half().float();ci=c.half().float();mi=z.half().float();ma=angles.half().float();pmean=mean.half().float();pb=basis.half().float();pcode=pc.half().float()
    methods={'independent_full_matrices':(mats.half().float(),[mats.half()],0.0),'tucker_free_coefficients':(torch.einsum('lk,kij->lij',ci,bi),[bank.half(),c.half()],0.0),'mirror_givens_coefficients':(torch.einsum('lk,kij->lij',torch.stack([rotate(mi,t) for t in ma]),bi),[bank.half(),z.half(),angles.half()],tm),'generic_pca_rank2_coefficients':(torch.einsum('lk,kij->lij',pcode@pb+pmean,bi),[bank.half(),mean.half(),basis.half(),pc.half()],tp)}
    for name,(pred,parts,sec) in methods.items():
     yy=torch.einsum('lbd,ldf->lbf',x,pred);payload=pack(name,parts,{'D':D,'bank_size':K,'layers':L});rows.append({'phase':phase,'world':w,'seed':seed,'stratum':st,'method':name,'matrix_nrmse':err(pred,mats),'functional_nrmse':err(yy,y),'payload_bytes':len(payload),'fit_seconds':sec,'payload_sha256':hashlib.sha256(payload).hexdigest()})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
