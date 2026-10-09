import csv,hashlib,io,json,math,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';DEV=[48400,48401];FRESH=[48410,48411,48412];SEEDS=[0,1,2];E=64;D=32;R=4;KS=[8,16,32,64]

def world(w,s):
 g=torch.Generator().manual_seed(w*100003+s*7919+41);base=torch.randn(D,generator=g)*.2;basis=torch.linalg.qr(torch.randn(D,R,generator=g)).Q
 coeff=torch.randn(E,R,generator=g)*.18
 # correlated coordinates make a shared functional subspace useful, but not discrete clusters
 coeff[:,1:]=.65*coeff[:,:1]+.35*coeff[:,1:]
 targets=base[None,:]+coeff@basis.T
 return base,basis,targets

def kmeans(x,k,seed):
 g=torch.Generator().manual_seed(seed);c=x[torch.randperm(len(x),generator=g)[:k]].clone()
 for _ in range(50):
  a=torch.cdist(x,c).argmin(1);n=c.clone()
  for j in range(k):
   m=a==j
   if m.any():n[j]=x[m].mean(0)
  if torch.allclose(c,n,atol=1e-7,rtol=0):break
  c=n
 return c

def fit_dev():
 out={}
 for k in KS:
  x=torch.cat([world(w,s)[2] for w in DEV for s in SEEDS]);out[str(k)]=kmeans(x,k,484001+k)
 torch.save(out,ART/'dev_codebooks.pt')

def serialize(method,base,basis,y,k,cb=None):
 if method=='independent':o={'method':method,'experts':y.clone(),'N':len(y)}
 elif method=='lowrank':o={'method':method,'base':base.clone(),'basis':basis.clone(),'coords':(y-base)@basis,'N':len(y)}
 elif method=='generic_vq':
  ids=torch.cdist(y,cb).argmin(1);o={'method':method,'codebook':cb.clone(),'ids':ids.to(torch.uint8),'N':len(y)}
 else:
  coords=(y-base)@basis;cc=None
  # Shared basis codebook is fitted and encoded strictly in the four-dimensional view coordinates.
  if cb is not None:cc=(cb-base)@basis
  ids=torch.cdist(coords,cc).argmin(1);o={'method':'mirror_vq','base':base.clone(),'basis':basis.clone(),'codebook':cc.clone(),'ids':ids.to(torch.uint8),'N':len(y)}
 b=io.BytesIO();torch.save(o,b);return b.getvalue()

def decode(o):
 m=o['method']
 if m=='independent':return o['experts']
 if m=='lowrank':return o['base'][None,:]+o['coords']@o['basis'].T
 if m=='generic_vq':return o['codebook'][o['ids'].long()]
 return o['base'][None,:]+o['codebook'][o['ids'].long()]@o['basis'].T

def run(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True)
 if phase=='development':fit_dev();print('development codebooks fitted');return
 books=torch.load(ART/'dev_codebooks.pt',weights_only=False);rows=[]
 for w in FRESH:
  for s in SEEDS:
   base,basis,y=world(w,s)
   for k in KS:
    for m in ['independent','lowrank','generic_vq','mirror_vq']:
     cb=books[str(k)];b=serialize(m,base,basis,y,k,cb);o=torch.load(io.BytesIO(b),weights_only=False);yh=decode(o);err=float((yh-y).norm()/(y.norm()+1e-12));uids=len(torch.unique(o.get('ids',torch.arange(E))));dup=0 if m in ('independent','lowrank') else 1-uids/E
     p=PAY/f'{w}_{s}_{m}_K{k}.pt';p.write_bytes(b);rows.append({'world':w,'seed':s,'method':m,'K':k,'normalized_rmse':err,'unique_codes':uids,'collision_fraction':dup,'payload_bytes':len(b),'bytes_per_expert':len(b)/E,'decode_MAC_per_expert':D*R if m in ('lowrank','mirror_vq') else 0,'hash':hashlib.sha256(b).hexdigest(),'path':str(p.relative_to(ROOT.parents[2]))})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=list(rows[0]));wri.writeheader();wri.writerows(rows)
 (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'rows':len(rows)}))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
