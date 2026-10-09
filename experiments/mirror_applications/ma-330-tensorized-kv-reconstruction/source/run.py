"""MA-330 tensorized KV cache reconstruction with per-role Mirror views."""
import argparse,csv,hashlib,json,struct,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];L,H,R,T,D=8,2,16,32,16;DEV=[33000,33001];FRESH=[33010,33011,33012];SEEDS=[0,1,2]
torch.set_num_threads(2)
def jrotate(x):
 q=x.reshape(*x.shape[:-1],-1,2);return torch.stack([-q[...,1],q[...,0]],-1).flatten(-2)
def rotate(x,t):
 q=x.reshape(*x.shape[:-1],-1,2);c=torch.cos(t);s=torch.sin(t);return torch.stack([c*q[...,0]-s*q[...,1],s*q[...,0]+c*q[...,1]],-1).flatten(-2)
def make(w,seed,stratum):
 g=torch.Generator().manual_seed(w*100003+seed*7919+330);r=R
 if stratum=='aligned':
  bk=torch.randn(T,D,generator=g);bv=torch.randn(T,D,generator=g);tk=torch.linspace(0,2*torch.pi,R);tv=torch.linspace(0,1.7*torch.pi,R);tk[0]=0;tv[0]=0;k=torch.stack([rotate(bk,t) for t in tk]);v=torch.stack([rotate(bv,t) for t in tv])
 else:k=torch.randn(R,T,D,generator=g);v=torch.randn(R,T,D,generator=g);bk=k[0].clone();bv=v[0].clone();tk=torch.zeros(R);tv=torch.zeros(R)
 q=torch.randn(R,8,D,generator=g);return k,v,q,bk,bv,tk,tv
def pack(name,parts):
 m={'method':name,'shapes':[list(t.shape) for t in parts],'dtypes':[str(t.dtype) for t in parts],'roles':R,'tokens':T};b=json.dumps(m,sort_keys=True,separators=(',',':')).encode();return b'MA330\0'+struct.pack('<I',len(b))+b+b''.join(t.contiguous().numpy().tobytes() for t in parts)
def nrmse(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-12))
def context(q,k,v):
 score=torch.einsum('rqd,rtd->rqt',q,k)/D**.5;return torch.einsum('rqt,rtd->rqd',torch.softmax(score,-1),v)
def fit_mirror(base,cache):
 jb=jrotate(base);den=base.square().sum().clamp_min(1e-9);jd=jb.square().sum().clamp_min(1e-9);angles=[]
 for row in cache:
  a=(row*base).sum()/den;b=(row*jb).sum()/jd;angles.append(torch.atan2(b,a))
 return torch.stack(angles)
def fit_pca(cache,rank=2):
 flat=cache.flatten(1);mean=flat.mean(0);u,s,vh=torch.linalg.svd(flat-mean,full_matrices=False);basis=vh[:rank];code=(flat-mean)@basis.T;return mean,basis,code
def run(phase):
 rows=[]
 for w in DEV if phase=='development' else FRESH:
  for seed in SEEDS:
   for st in ('aligned','independent'):
    k,v,q,bk,bv,tk,tv=make(w,seed,st);target=context(q,k,v)
    t0=time.perf_counter();mk=fit_mirror(bk,k);mv=fit_mirror(bv,v);me=time.perf_counter()-t0;t0=time.perf_counter();mdk=torch.stack([rotate(bk.half().float(),a.half().float()) for a in mk]);mdv=torch.stack([rotate(bv.half().float(),a.half().float()) for a in mv]);tm=time.perf_counter()-t0
    t0=time.perf_counter();kmean,kbasis,kcode=fit_pca(k);vmean,vbasis,vcode=fit_pca(v);pe=time.perf_counter()-t0;t0=time.perf_counter();pk=(kcode.half().float()@kbasis.half().float()+kmean.half().float()).reshape(R,T,D);pv=(vcode.half().float()@vbasis.half().float()+vmean.half().float()).reshape(R,T,D);tp=time.perf_counter()-t0
    sharedk=k.mean(0).half().float().expand(R,T,D);sharedv=v.mean(0).half().float().expand(R,T,D);basek=bk.half().float();basev=bv.half().float()
    methods={'full_independent_kv':(k.half().float(),v.half().float(),[k.half(),v.half()],0.0,0.0),'mlkv_single_shared_state':(sharedk,sharedv,[k.mean(0).half(),v.mean(0).half()],0.0,0.0),'mirror_role_views':(mdk,mdv,[basek.half(),basev.half(),mk.half(),mv.half()],tm,me),'generic_tucker_pca_rank2':(pk,pv,[kmean.half(),kbasis.half(),kcode.half(),vmean.half(),vbasis.half(),vcode.half()],tp,pe)}
    for name,(kr,vr,parts,secs,encsecs) in methods.items():
     out=context(q,kr,vr);payload=pack(name,parts);rows.append({'phase':phase,'world':w,'seed':seed,'stratum':st,'method':name,'kv_nrmse':(nrmse(kr,k)+nrmse(vr,v))/2,'attention_context_nrmse':nrmse(out,target),'payload_bytes':len(payload),'encode_seconds':encsecs,'materialize_seconds':secs,'cache_roles':R,'cached_tokens':R*T,'materialization_macs':(0 if name in ('full_independent_kv','mlkv_single_shared_state') else R*T*D*4),'payload_sha256':hashlib.sha256(payload).hexdigest()})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
