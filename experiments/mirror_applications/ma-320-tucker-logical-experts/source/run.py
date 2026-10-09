"""MA-320 Tucker bank for logical MoE experts with Mirror coefficient views."""
import argparse,csv,hashlib,json,struct,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];D,K,E=16,8,128;DEV=[32000,32001];FRESH=[32010,32011,32012];SEEDS=[0,1,2]
torch.set_num_threads(2)
def rotate(z,t):
 q=z.reshape(-1,2);c=torch.cos(t);s=torch.sin(t);return torch.stack([c*q[:,0]-s*q[:,1],s*q[:,0]+c*q[:,1]],-1).reshape(-1)
def make(w,seed,stratum):
 g=torch.Generator().manual_seed(w*100003+seed*7919+320);bank=torch.randn(K,D,D,generator=g);bank=bank/bank.flatten(1).norm(dim=1)[:,None,None]
 if stratum=='aligned':
  z=torch.randn(K,generator=g);theta=torch.linspace(0,2*torch.pi,E);theta[0]=0;coef=torch.stack([rotate(z,t) for t in theta])
 else:coef=torch.randn(E,K,generator=g);z=coef[0].clone();theta=torch.zeros(E)
 experts=torch.einsum('ek,kij->eij',coef,bank)
 # Fixed deterministic top-1 dispatch schedule shared by all methods.
 dispatch=(torch.arange(E*32)*37% E);x=torch.randn(E*32,D,generator=g);y=torch.einsum('td,tdh->th',x,experts[dispatch]);return bank,coef,z,theta,experts,x,y,dispatch
def pack(name,parts):
 m={'method':name,'shapes':[list(t.shape) for t in parts],'dtypes':[str(t.dtype) for t in parts],'dispatch':'fixed round-robin hash; no learned router'};b=json.dumps(m,sort_keys=True,separators=(',',':')).encode();return b'MA320\0'+struct.pack('<I',len(b))+b+b''.join(t.contiguous().numpy().tobytes() for t in parts)
def err(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-12))
def run(phase):
 rows=[]
 for w in DEV if phase=='development' else FRESH:
  for seed in SEEDS:
   for st in ('aligned','independent'):
    bank,c,z,theta,experts,x,y,dispatch=make(w,seed,st);t0=time.perf_counter();Jz=torch.zeros_like(z);q=z.reshape(-1,2);Jz.reshape(-1,2)[:,0]=-q[:,1];Jz.reshape(-1,2)[:,1]=q[:,0];angles=[]
    for row in c:
     a=row.dot(z)/z.dot(z).clamp_min(1e-9);b=row.dot(Jz)/Jz.dot(Jz).clamp_min(1e-9);angles.append(torch.atan2(b,a))
    angles=torch.stack(angles);mc=torch.stack([rotate(z,t) for t in angles]);mirror=torch.einsum('ek,kij->eij',mc,bank);tm=time.perf_counter()-t0
    t0=time.perf_counter();mean=c.mean(0);u,s,vh=torch.linalg.svd(c-mean,full_matrices=False);basis=vh[:2];pc=(c-mean)@basis.T;pca_c=pc@basis+mean;pca=torch.einsum('ek,kij->eij',pca_c,bank);tp=time.perf_counter()-t0
    bi=bank.half().float();methods={'independent_full_experts':(experts.half().float(),[experts.half()],0.0),'tucker_free_experts':(torch.einsum('ek,kij->eij',c.half().float(),bi),[bank.half(),c.half()],0.0),'mirror_expert_views':(torch.einsum('ek,kij->eij',torch.stack([rotate(z.half().float(),t) for t in angles.half().float()]),bi),[bank.half(),z.half(),angles.half()],tm),'generic_pca_experts':(torch.einsum('ek,kij->eij',(pc.half().float()@basis.half().float())+mean.half().float(),bi),[bank.half(),mean.half(),basis.half(),pc.half()],tp)}
    for name,(pred,parts,secs) in methods.items():
     out=torch.einsum('td,tdh->th',x,pred[dispatch]);payload=pack(name,parts);rows.append({'phase':phase,'world':w,'seed':seed,'stratum':st,'method':name,'expert_matrix_nrmse':err(pred,experts),'routed_output_nrmse':err(out,y),'payload_bytes':len(payload),'fit_seconds':secs,'dispatch_tokens':int(dispatch.numel()),'expert_macs_per_token':D*D,'one_time_materialization_macs':(0 if name=='independent_full_experts' else E*K*D*D),'payload_sha256':hashlib.sha256(payload).hexdigest()})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
