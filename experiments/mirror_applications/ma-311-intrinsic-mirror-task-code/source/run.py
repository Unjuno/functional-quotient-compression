"""MA-311 Mirror views over a random intrinsic task subspace."""
import argparse,csv,hashlib,json,struct,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];D,d,N=128,16,8;DEV=[31100,31101];FRESH=[31110,31111,31112];SEEDS=[0,1,2];UPDATES=800
torch.set_num_threads(2)
def rotate(z,theta):
 q=z.reshape(-1,2);c=torch.cos(theta);s=torch.sin(theta);return torch.stack([c*q[:,0]-s*q[:,1],s*q[:,0]+c*q[:,1]],-1).reshape(-1)
def make(w,seed,stratum):
 g=torch.Generator().manual_seed(w*100003+seed*7919+311);U=torch.linalg.qr(torch.randn(D,d,generator=g)).Q[:,:d]
 htr=torch.randn(N,128,d,generator=g);hq=torch.randn(N,256,d,generator=g)
 if stratum=='aligned':
  z=torch.randn(d,generator=g);theta=torch.linspace(0,2.2,N);theta[0]=0;coef=torch.stack([rotate(z,t) for t in theta])
 else:coef=torch.randn(N,d,generator=g);theta=torch.zeros(N);z=coef[0].clone()
 ytr=torch.einsum('ntd,nd->nt',htr,coef);yq=torch.einsum('ntd,nd->nt',hq,coef)
 return U,htr,ytr,hq,yq,coef,z,theta
def pack(name,parts,meta):
 md={'method':name,'shapes':[list(t.shape) for t in parts],'dtypes':[str(t.dtype) for t in parts]};md.update(meta);b=json.dumps(md,sort_keys=True,separators=(',',':')).encode();return b'MA311\0'+struct.pack('<I',len(b))+b+b''.join(t.contiguous().numpy().tobytes() for t in parts)
def nrmse(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-12))
def run(phase):
 rows=[]
 for world in DEV if phase=='development' else FRESH:
  for seed in SEEDS:
   for stratum in ('aligned','independent'):
    U,htr,ytr,hq,yq,teacher,ztrue,thetatrue=make(world,seed,stratum)
    # Mandatory random intrinsic subspace control: task coefficients fit independently by support least squares.
    t0=time.perf_counter();direct=torch.stack([torch.linalg.lstsq(htr[i],ytr[i]).solution for i in range(N)]);direct_s=time.perf_counter()-t0
    # Mirror: one shared intrinsic vector and per-task Givens angle, learned jointly from task support.
    mz=torch.nn.Parameter(direct[0].detach().clone());mt=torch.nn.Parameter(torch.zeros(N));opt=torch.optim.Adam([mz,mt],lr=.03);t0=time.perf_counter()
    for _ in range(UPDATES):
     opt.zero_grad();pred=torch.stack([rotate(mz,mt[i]) for i in range(N)]);loss=torch.einsum('ntd,nd->nt',htr,pred)-ytr;loss=loss.square().mean();loss.backward();opt.step()
    mirror=torch.stack([rotate(mz.detach(),mt.detach()[i]) for i in range(N)]);mirror_s=time.perf_counter()-t0
    # Generic rank-2 PCA over independently fitted intrinsic task vectors.
    t0=time.perf_counter();mean=direct.mean(0);u,sv,vh=torch.linalg.svd(direct-mean,full_matrices=False);basis=vh[:2];pc=(direct-mean)@basis.T;pca=pc@basis+mean;pca_s=time.perf_counter()-t0
    methods={'random_intrinsic_independent':(direct,[U.half(),direct.half()],direct_s,0),'mirror_intrinsic_orbit':(mirror,[U.half(),mz.detach().half(),mt.detach().half()],mirror_s,UPDATES),'generic_intrinsic_pca_rank2':(pca,[U.half(),mean.half(),basis.half(),pc.half()],pca_s,0)}
    for name,(coef,parts,sec,updates) in methods.items():
     out=torch.einsum('ntd,nd->nt',hq,coef);payload=pack(name,parts,{'D':D,'intrinsic_dim':d,'tasks':N});rows.append({'phase':phase,'world':world,'seed':seed,'stratum':stratum,'method':name,'query_nrmse':nrmse(out,yq),'payload_bytes':len(payload),'fit_seconds':sec,'optimizer_updates':updates,'examples_seen':N*128*(updates if updates else 1),'payload_sha256':hashlib.sha256(payload).hexdigest()})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
