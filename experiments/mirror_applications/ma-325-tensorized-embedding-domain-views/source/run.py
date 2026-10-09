"""MA-325 tensorized embedding domains using shared TT cores and a Mirror view."""
import argparse,csv,hashlib,json,struct,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];V,ED,DOM=256,256,32;DEV=[32500,32501];FRESH=[32510,32511,32512];SEEDS=[0,1,2]
torch.set_num_threads(2)
def rot(t):return torch.stack([torch.stack([torch.cos(t),-torch.sin(t)]),torch.stack([torch.sin(t),torch.cos(t)])])
def mod_core(c,t):return torch.einsum('ab,bijk->aijk',rot(t),c)
def tt_embedding(cores):
 a,b,c,d=cores;t=torch.tensordot(a[0],b,dims=([-1],[0]));t=torch.tensordot(t,c,dims=([-1],[0]));t=torch.tensordot(t,d,dims=([-1],[0]));t=t.squeeze(-1);t=t.permute(0,2,4,6,1,3,5,7).contiguous();return t.reshape(V,ED)
def make(w,seed,stratum):
 g=torch.Generator().manual_seed(w*100003+seed*7919+325);g1=torch.randn(1,4,4,2,generator=g);g2=torch.randn(2,4,4,2,generator=g);g3=torch.randn(2,4,4,2,generator=g);g4=torch.randn(2,4,4,1,generator=g)
 if stratum=='aligned':theta=torch.linspace(0,2*torch.pi,DOM);theta[0]=0;domains=torch.stack([mod_core(g2,t) for t in theta])
 else:theta=torch.zeros(DOM);domains=torch.randn(DOM,2,4,4,2,generator=g)
 embeddings=torch.stack([tt_embedding((g1,domains[i],g3,g4)) for i in range(DOM)]);tokens=torch.randint(V,(DOM,64),generator=g);return (g1,g2,g3,g4),domains,theta,embeddings,tokens
def pack(name,parts):
 md={'method':name,'shapes':[list(t.shape) for t in parts],'dtypes':[str(t.dtype) for t in parts]};b=json.dumps(md,sort_keys=True,separators=(',',':')).encode();return b'MA325\0'+struct.pack('<I',len(b))+b+b''.join(t.contiguous().numpy().tobytes() for t in parts)
def err(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-12))
def run(phase):
 rows=[]
 for w in DEV if phase=='development' else FRESH:
  for seed in SEEDS:
   for st in ('aligned','independent'):
    base,cores,theta,target,tokens=make(w,seed,st);g1,g2,g3,g4=base;t0=time.perf_counter();v=g2.reshape(2,-1);j=torch.stack([-v[1],v[0]]);z=v.flatten();jz=j.flatten();angles=[]
    for c in cores:
     q=c.reshape(2,-1).flatten();a=q.dot(z)/z.dot(z).clamp_min(1e-9);b=q.dot(jz)/jz.dot(jz).clamp_min(1e-9);angles.append(torch.atan2(b,a))
    angles=torch.stack(angles);tm=time.perf_counter()-t0;t0=time.perf_counter();flat=cores.flatten(1);mean=flat.mean(0);u,s,vh=torch.linalg.svd(flat-mean,full_matrices=False);basis=vh[:2];pc=(flat-mean)@basis.T;pca_core=(pc@basis+mean).reshape_as(cores);tp=time.perf_counter()-t0
    g1d,g2d,g3d,g4d=[q.half().float() for q in base]
    methods={'independent_full_embeddings':(target.half().float(),[target.half()],0.0),'tt_independent_domain_core':(torch.stack([tt_embedding((g1d,cores[i].half().float(),g3d,g4d)) for i in range(DOM)]),[g1.half(),g3.half(),g4.half(),cores.half()],0.0),'mirror_domain_view':(torch.stack([tt_embedding((g1d,mod_core(g2d,angles[i].half().float()),g3d,g4d)) for i in range(DOM)]),[g1.half(),g2.half(),g3.half(),g4.half(),angles.half()],tm),'generic_pca_domain_core':(torch.stack([tt_embedding((g1d,pca_core[i].half().float(),g3d,g4d)) for i in range(DOM)]),[g1.half(),g3.half(),g4.half(),mean.half(),basis.half(),pc.half()],tp)}
    for name,(pred,parts,secs) in methods.items():
     predq=torch.stack([pred[i,tokens[i]] for i in range(DOM)]);targetq=torch.stack([target[i,tokens[i]] for i in range(DOM)]);payload=pack(name,parts);rows.append({'phase':phase,'world':w,'seed':seed,'stratum':st,'method':name,'embedding_nrmse':err(pred,target),'lookup_nrmse':err(predq,targetq),'payload_bytes':len(payload),'fit_seconds':secs,'token_lookups':int(tokens.numel()),'payload_sha256':hashlib.sha256(payload).hexdigest()})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
