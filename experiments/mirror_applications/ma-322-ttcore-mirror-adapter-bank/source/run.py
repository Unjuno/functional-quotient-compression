"""MA-322 shared Tensor-Train adapter cores with a Mirror view on one core."""
import argparse,csv,hashlib,json,struct,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];D,E=16,64;DEV=[32200,32201];FRESH=[32210,32211,32212];SEEDS=[0,1,2]
torch.set_num_threads(2)
def rot2(t):return torch.stack([torch.stack([torch.cos(t),-torch.sin(t)]),torch.stack([torch.sin(t),torch.cos(t)])])
def rotate_core(core,t):return torch.einsum('ab,bijk->aijk',rot2(t),core)
def tt_matrix(cores):
 a,b,c,d=cores;t=torch.tensordot(a[0],b,dims=([-1],[0]));t=torch.tensordot(t,c,dims=([-1],[0]));t=torch.tensordot(t,d,dims=([-1],[0]));t=t.squeeze(-1);t=t.permute(0,2,4,6,1,3,5,7).contiguous();return t.reshape(D,D)
def make(w,seed,stratum):
 g=torch.Generator().manual_seed(w*100003+seed*7919+322);g1=torch.randn(1,2,2,2,generator=g);g2=torch.randn(2,2,2,2,generator=g);g3=torch.randn(2,2,2,2,generator=g);g4=torch.randn(2,2,2,1,generator=g)
 if stratum=='aligned':theta=torch.linspace(0,2*torch.pi,E);theta[0]=0;cores2=torch.stack([rotate_core(g2,t) for t in theta])
 else:theta=torch.zeros(E);cores2=torch.randn(E,2,2,2,2,generator=g)
 x=torch.randn(E,64,D,generator=g);return (g1,g2,g3,g4),cores2,theta,x
def pack(name,parts):
 m={'method':name,'shapes':[list(t.shape) for t in parts],'dtypes':[str(t.dtype) for t in parts]};b=json.dumps(m,sort_keys=True,separators=(',',':')).encode();return b'MA322\0'+struct.pack('<I',len(b))+b+b''.join(t.contiguous().numpy().tobytes() for t in parts)
def nrmse(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-12))
def run(phase):
 rows=[]
 for w in DEV if phase=='development' else FRESH:
  for seed in SEEDS:
   for st in ('aligned','independent'):
    base,cc,theta,x=make(w,seed,st);g1,g2,g3,g4=base;target_mats=torch.stack([tt_matrix((g1,cc[i],g3,g4)) for i in range(E)]);y=torch.einsum('etd,edh->eth',x,target_mats)
    t0=time.perf_counter();v=g2.reshape(2,-1);j=torch.stack([-v[1],v[0]]);zv=v.flatten();jv=j.flatten();angles=[]
    for c in cc:
     q=c.reshape(2,-1).flatten();a=q.dot(zv)/zv.dot(zv).clamp_min(1e-9);b=q.dot(jv)/jv.dot(jv).clamp_min(1e-9);angles.append(torch.atan2(b,a))
    angles=torch.stack(angles);mi=torch.stack([rotate_core(g2,t) for t in angles]);mirror_mats=torch.stack([tt_matrix((g1,mi[i],g3,g4)) for i in range(E)]);tm=time.perf_counter()-t0
    t0=time.perf_counter();flat=cc.flatten(1);mean=flat.mean(0);u,s,vh=torch.linalg.svd(flat-mean,full_matrices=False);basis=vh[:2];pc=(flat-mean)@basis.T;pflat=pc@basis+mean;generic_cores=pflat.reshape_as(cc);g1d,g2d,g3d,g4d=[q.half().float() for q in base];pca_mats=torch.stack([tt_matrix((g1d,generic_cores[i].half().float(),g3d,g4d)) for i in range(E)]);tp=time.perf_counter()-t0
    g1d,g2d,g3d,g4d=[q.half().float() for q in base];qbase=[g1d,g3d,g4d];bi=[g1d,g2d,g3d,g4d]
    methods={'independent_full_adapter':(target_mats.half().float(),[target_mats.half()],0.0),'loretta_independent_tt_core':(torch.stack([tt_matrix((g1d,cc[i].half().float(),g3d,g4d)) for i in range(E)]),qbase+[cc.half()],0.0),'mirror_tt_core_view':(torch.stack([tt_matrix((g1d,rotate_core(g2d,angles[i].half().float()),g3d,g4d)) for i in range(E)]),bi+[angles.half()],tm),'generic_pca_tt_core':(pca_mats,[g1.half(),g3.half(),g4.half(),mean.half(),basis.half(),pc.half()],tp)}
    for name,(pred,parts,secs) in methods.items():
     out=torch.einsum('etd,edh->eth',x,pred);payload=pack(name,parts);rows.append({'phase':phase,'world':w,'seed':seed,'stratum':st,'method':name,'matrix_nrmse':nrmse(pred,target_mats),'functional_nrmse':nrmse(out,y),'payload_bytes':len(payload),'fit_seconds':secs,'payload_sha256':hashlib.sha256(payload).hexdigest()})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
