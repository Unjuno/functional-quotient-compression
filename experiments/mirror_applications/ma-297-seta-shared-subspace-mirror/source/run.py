"""MA-297 SETA-inspired shared/private sparse task screen."""
import argparse,csv,hashlib,json,struct,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];D,N,S,R=128,8,16,3;DEV=[29700,29701];FRESH=[29710,29711,29712];SEEDS=[0,1,2]
torch.set_num_threads(2)
def make(w,s):
 g=torch.Generator().manual_seed(w*100003+s*7919+297);perm=torch.randperm(D,generator=g);shared=perm[:S]; priv=perm[S:S+N*4].reshape(N,4)
 u=torch.randn(S,R,generator=g);c=torch.randn(N,R,generator=g);mat=c@u.T
 ds=torch.zeros(N,D);ds[:,shared]=mat
 pv=torch.randn(N,4,generator=g);ds.scatter_(1,priv,pv)
 x=torch.randn(N,640,D,generator=g);y=torch.einsum('ntd,nd->nt',x,ds)
 return ds,shared,priv,x,y

def pack(parts,name):
 meta=json.dumps({'method':name,'shapes':[list(p.shape) for p in parts],'dtype':'float32'},sort_keys=True,separators=(',',':')).encode();raw=b''.join(p.contiguous().numpy().tobytes() for p in parts);return b'MA297\0'+struct.pack('<I',len(meta))+meta+raw
def err(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-12))
def run(phase):
 rows=[]
 for w in DEV if phase=='development' else FRESH:
  for seed in SEEDS:
   ds,shared,priv,x,y=make(w,seed);start=time.perf_counter()
   # Discover shared support only from first six tasks via repeated coordinate occupancy.
   freq=(ds[:6].abs()>1e-7).sum(0); discovered=torch.topk(freq, S).indices
   actual=set(shared.tolist()); discovered_ok=len(actual.intersection(discovered.tolist()))==S
   mat=ds[:,discovered];u,sv,vh=torch.linalg.svd(mat,full_matrices=False);r=min(R,sv.numel());basis=vh[:r];codes=mat@basis.T;recon=codes@basis
   out=torch.zeros_like(ds);out[:,discovered]=recon
   # Private residuals are paid in all shared/private methods.
   residual=ds.clone();residual[:,discovered]=0;out+=residual
   ri=torch.nonzero(ds!=0,as_tuple=False).to(torch.float32); rv=ds[ds!=0]
   rsi=torch.nonzero(residual!=0,as_tuple=False).to(torch.float32); rsv=residual[residual!=0]
   shidx=discovered.to(torch.float32)
   candidates={'independent_sparse':(ds,[ri,rv]),'seta_shared_private':(ds,[shidx,mat,rsi,rsv]),'mirror_shared_code':(out,[shidx,basis,codes,rsi,rsv]),'generic_pca_shared_code':(out,[shidx,basis,codes,rsi,rsv])}
   for name,(pred,parts) in candidates.items():
    blob=pack(parts,name);qpred=torch.einsum('ntd,nd->nt',x[:,512:],pred);qtrue=y[:,512:]
    rows.append({'phase':phase,'world':w,'seed':seed,'method':name,'support_recall':int(discovered_ok),'nrmse':err(pred,ds),'query_nrmse':err(qpred,qtrue),'payload_bytes':len(blob),'wall_seconds':time.perf_counter()-start,'sha256':hashlib.sha256(blob).hexdigest()})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
