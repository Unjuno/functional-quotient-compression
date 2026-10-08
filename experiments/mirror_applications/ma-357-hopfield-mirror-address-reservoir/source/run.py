#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';SEEDS=(35701,35702,35703,35711,35712,35713);KS=(16,64,256);BITS=32;CODE=8;NQ=2000

def pack(state,meta):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for k,a in sorted(state.items()):
   q=io.BytesIO();np.save(q,np.asarray(a),allow_pickle=False);i=zipfile.ZipInfo(k+'.npy',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,q.getvalue())
  i=zipfile.ZipInfo('metadata.json',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,json.dumps(meta,sort_keys=True,separators=(',',':')).encode())
 p=b.getvalue()
 with zipfile.ZipFile(io.BytesIO(p)) as z:s={n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')};m=json.loads(z.read('metadata.json'))
 return p,s,m

def world(seed,k):
 r=np.random.default_rng(seed+k*97);patterns=r.integers(0,2,size=(k,BITS),dtype=np.uint8);patterns[:,0]=np.arange(k)%2
 # Ensure unique keys (with overwhelming probability, but enforce deterministically).
 seen=set()
 for i in range(k):
  while patterns[i].tobytes() in seen:patterns[i]=r.integers(0,2,size=BITS,dtype=np.uint8)
  seen.add(patterns[i].tobytes())
 codes=r.normal(size=(k,CODE)).astype('f4')*.2;codes/=np.maximum(np.linalg.norm(codes,axis=1,keepdims=True),1e-6)
 qidx=r.integers(k,size=NQ);flips=r.integers(0,9,size=NQ);queries=patterns[qidx].copy()
 for n,b in enumerate(flips):
  if b:
   jj=r.choice(BITS,size=b,replace=False);queries[n,jj]^=1
 return patterns,codes,qidx,flips,queries

def run(seed,k,method):
 patterns,codes,qidx,flips,queries=world(seed,k);state={'addresses_packed':np.packbits(patterns,axis=1),'view_codes':codes,'temperature':np.array([8.],'f4')};meta={'method':method,'K':k,'bits':BITS,'code_dim':CODE,'format':'MA357-v1'}
 start=time.perf_counter();p,s,m=pack(state,meta);stored=np.unpackbits(s['addresses_packed'],axis=1)[:,:BITS]
 d=(queries[:,None,:]!=stored[None,:,:]).sum(2);scores=BITS-2*d
 if method=='explicit_hamming':idx=np.argmin(d,axis=1);approx=s['view_codes'][idx]
 elif method=='modern_hopfield':
  logits=scores.astype('f4')*float(s['temperature'][0])/BITS;logits-=logits.max(1,keepdims=True);weights=np.exp(logits);weights/=weights.sum(1,keepdims=True);approx=weights@s['view_codes'];idx=np.argmax(weights,axis=1)
 elif method=='oracle_exact':idx=qidx;approx=s['view_codes'][idx]
 else:raise ValueError(method)
 acc=float(np.mean(idx==qidx));nMSE=float(np.mean((approx-codes[qidx])**2)/(np.mean(codes[qidx]**2)+1e-12));collisions=1-len({r.tobytes() for r in stored})/k
 macs=k*BITS + (k*CODE if method=='modern_hopfield' else 0)
 return {'seed':seed,'K':k,'method':method,'payload_bytes':len(p),'sha256':hashlib.sha256(p).hexdigest(),'query_accuracy':acc,'decoded_code_nMSE':nMSE,'collision_rate':collisions,'mean_corrupt_bits':float(flips.mean()),'queries':NQ,'optimizer_updates':0,'lookup_MAC_proxy':int(macs),'wall_seconds':time.perf_counter()-start}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dev-only',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True);rows=[]
 for seed in SEEDS[:3] if args.dev_only else SEEDS:
  for k in KS:
   for method in ('oracle_exact','explicit_hamming','modern_hopfield'):
    a=run(seed,k,method);rows.append(a);print(json.dumps(a,sort_keys=True))
 p=OUT/('development.csv' if args.dev_only else 'results.csv')
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
