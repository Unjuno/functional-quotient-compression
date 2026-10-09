"""MA-312 scaling intrinsic Mirror task codes to many tasks."""
import argparse,csv,hashlib,json,struct,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];D,d,N=256,32,64;DEV=[31200,31201];FRESH=[31210,31211,31212];SEEDS=[0,1,2];UPDATES=800
torch.set_num_threads(2)
def rotate_batch(z,theta):
 q=z.reshape(1,-1,2);c=torch.cos(theta).unsqueeze(1);s=torch.sin(theta).unsqueeze(1);return torch.stack([c*q[...,0]-s*q[...,1],s*q[...,0]+c*q[...,1]],-1).reshape(theta.shape[0],-1)
def make(w,seed,stratum):
 g=torch.Generator().manual_seed(w*100003+seed*7919+312);U=torch.linalg.qr(torch.randn(D,d,generator=g)).Q[:,:d];htr=torch.randn(N,64,d,generator=g);hq=torch.randn(N,128,d,generator=g)
 if stratum=='aligned':
  z=torch.randn(d,generator=g);theta=torch.linspace(0,2*torch.pi,N);theta[0]=0;coef=rotate_batch(z,theta)
 else:coef=torch.randn(N,d,generator=g);theta=torch.zeros(N);z=coef[0].clone()
 ytr=torch.einsum('ntd,nd->nt',htr,coef);yq=torch.einsum('ntd,nd->nt',hq,coef);return U,htr,ytr,hq,yq,coef
def pack(name,parts,meta):
 md={'method':name,'shapes':[list(t.shape) for t in parts],'dtypes':[str(t.dtype) for t in parts]};md.update(meta);b=json.dumps(md,sort_keys=True,separators=(',',':')).encode();return b'MA312\0'+struct.pack('<I',len(b))+b+b''.join(t.contiguous().numpy().tobytes() for t in parts)
def err(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-12))
def run(phase):
 rows=[]
 for w in DEV if phase=='development' else FRESH:
  for seed in SEEDS:
   for st in ('aligned','independent'):
    U,htr,ytr,hq,yq,teacher=make(w,seed,st);t0=time.perf_counter();direct=torch.stack([torch.linalg.lstsq(htr[i],ytr[i]).solution for i in range(N)]);td=time.perf_counter()-t0
    z=torch.nn.Parameter(direct[0].detach().clone());theta=torch.nn.Parameter(torch.zeros(N));opt=torch.optim.Adam([z,theta],lr=.03);t0=time.perf_counter()
    for _ in range(UPDATES):
     opt.zero_grad();pred=rotate_batch(z,theta);loss=(torch.einsum('ntd,nd->nt',htr,pred)-ytr).square().mean();loss.backward();opt.step()
    mirror=rotate_batch(z.detach(),theta.detach());tm=time.perf_counter()-t0
    t0=time.perf_counter();mean=direct.mean(0);u,s,vh=torch.linalg.svd(direct-mean,full_matrices=False);basis=vh[:2];pc=(direct-mean)@basis.T;pca=pc@basis+mean;tp=time.perf_counter()-t0
    items={'random_intrinsic_independent':(direct,[U.half(),direct.half()],[direct.half()],td,0),'mirror_many_codes':(mirror,[U.half(),z.detach().half(),theta.detach().half()],[z.detach().half(),theta.detach().half()],tm,UPDATES),'generic_intrinsic_pca_rank2':(pca,[U.half(),mean.half(),basis.half(),pc.half()],[mean.half(),basis.half(),pc.half()],tp,0)}
    for name,(coef,totalparts,adaptparts,secs,updates) in items.items():
     out=torch.einsum('ntd,nd->nt',hq,coef);full=pack(name,totalparts,{'D':D,'d':d,'N':N});adapt=pack(name+'_adapt',adaptparts,{'N':N});rows.append({'phase':phase,'world':w,'seed':seed,'stratum':st,'method':name,'query_nrmse':err(out,yq),'total_payload_bytes':len(full),'task_adaptation_bytes':len(adapt),'fit_seconds':secs,'optimizer_updates':updates,'examples_seen':N*64*(updates if updates else 1),'payload_sha256':hashlib.sha256(full).hexdigest()})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
