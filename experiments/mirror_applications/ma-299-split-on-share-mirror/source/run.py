"""MA-299 sequential split-on-share storage screen."""
import argparse,csv,hashlib,json,struct
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];D,N,R=256,10,3;DEV=[29900,29901];FRESH=[29910,29911,29912];SEEDS=[0,1,2];THRESH=.15
torch.set_num_threads(2)
def make(w,s):
 g=torch.Generator().manual_seed(w*100003+s*7919+299);basis=torch.randn(D,R,generator=g);basis=torch.linalg.qr(basis).Q[:,:R];coef=torch.randn(N,R,generator=g);ds=coef@basis.T
 # Later tasks introduce sparse private coordinates, triggering physical split.
 for i in range(6,N):
  idx=torch.randperm(D,generator=g)[:8];ds[i,idx]+=torch.randn(8,generator=g)*2
 x=torch.randn(N,128,D,generator=g);y=torch.einsum('ntd,nd->nt',x,ds);return ds,x,y
def pack(parts,name):
 meta=json.dumps({'method':name,'shapes':[list(p.shape) for p in parts],'dtypes':[str(p.dtype) for p in parts]},sort_keys=True,separators=(',',':')).encode();return b'MA299\0'+struct.pack('<I',len(meta))+meta+b''.join(p.contiguous().numpy().tobytes() for p in parts)
def run(phase):
 rows=[]
 for w in DEV if phase=='development' else FRESH:
  for seed in SEEDS:
   ds,x,y=make(w,seed);u,sv,vh=torch.linalg.svd(ds[:6],full_matrices=False);basis=vh[:R].T;codes=ds@basis;recon=codes@basis.T;res=ds-recon;rel=res.norm(dim=1)/ds.norm(dim=1).clamp_min(1e-12);split=rel>THRESH
   # Sparse index/value payload for private residuals after split.
   def sparse(z):
    nz=torch.nonzero(z!=0,as_tuple=False).float();val=z[z!=0];return nz,val
   stored=torch.where(res.abs()>1e-6,res,torch.zeros_like(res));idx,val=sparse(stored*split[:,None]);rawidx,rawval=sparse(ds);idx=idx.to(torch.int16);rawidx=rawidx.to(torch.int16)
   methods={'independent_private':(ds,[rawidx,rawval]),'never_split':(recon,[basis,codes]),'adaptive_mirror_split':(recon+res*split[:,None],[basis,codes,idx,val]),'generic_pca_split':(recon+res*split[:,None],[basis,codes,idx,val])}
   for name,(pred,parts) in methods.items():
    blob=pack(parts,name);q=torch.einsum('ntd,nd->nt',x,pred);rows.append({'phase':phase,'world':w,'seed':seed,'method':name,'split_count':int(split.sum()) if name in ('adaptive_mirror_split','generic_pca_split') else (0 if name=='never_split' else N),'novel_split_count':int(split[6:].sum()) if name in ('adaptive_mirror_split','generic_pca_split') else (0 if name=='never_split' else N),'query_nrmse':float((q-y).norm()/y.norm()),'payload_bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
