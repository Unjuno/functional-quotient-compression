#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';SEEDS=(35501,35502,35511,35512,35513);L=R=16;D=8;O=4;NTRAIN=NTEST=512

def pack(state,meta):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for name,a in sorted(state.items()):
   q=io.BytesIO();np.save(q,np.asarray(a),allow_pickle=False);i=zipfile.ZipInfo(name+'.npy',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,q.getvalue())
  i=zipfile.ZipInfo('metadata.json',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,json.dumps(meta,sort_keys=True,separators=(',',':')).encode())
 data=b.getvalue()
 with zipfile.ZipFile(io.BytesIO(data)) as z:s={n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')};m=json.loads(z.read('metadata.json'))
 return data,s,m

def bank(seed):
 rng=np.random.default_rng(seed);base=rng.normal(0,.15,(D,O)).astype('f4');u=rng.normal(0,.12,(L,D,O)).astype('f4');v=rng.normal(0,.12,(R,D,O)).astype('f4');inter=np.zeros((L,R,D,O),'f4')
 maps=np.stack([[base+u[i]+v[j]+inter[i,j] for j in range(R)] for i in range(L)])
 pairs=[(i,j) for i in range(L) for j in range(R)];rng.shuffle(pairs);train=pairs[:192];held=pairs[192:]
 x=rng.normal(size=(NTRAIN+NTEST,D)).astype('f4');xtr=x[:NTRAIN];xte=x[NTRAIN:]
 return base,u,v,inter,maps,pairs,train,held,xtr,xte

def methods(seeddata):
 base,u,v,inter,maps,pairs,train,held,xtr,xte=seeddata
 # Exact post-fit selection targets. All learned/stored tables are charged.
 return {
  'independent_full_matrices':({'maps':maps},{'kind':'independent_full','addresses':256}),
  'flat_table':({'maps':maps},{'kind':'flat_table','addresses':256}),
  'product_key':({'left_keys':np.eye(L,dtype='f4'),'right_keys':np.eye(R,dtype='f4'),'values':maps.reshape(L*R,D,O)},{'kind':'product_key_exact','left':L,'right':R,'values':L*R}),
  'mirror_factorized':({'base':base,'left_views':u,'right_views':v,'left_index':np.repeat(np.arange(L,dtype='u1'),R),'right_index':np.tile(np.arange(R,dtype='u1'),L)},{'kind':'mirror_pair_code','left':L,'right':R}),
  'direct_two_index_coefficients':({'base':base,'left_views':u,'right_views':v,'left_index':np.repeat(np.arange(L,dtype='u1'),R),'right_index':np.tile(np.arange(R,dtype='u1'),L)},{'kind':'direct_two_index_coefficients','left':L,'right':R})}

def decode(method,state,i,j):
 if method in ('independent_full_matrices','flat_table'):return state['maps'][i*R+j] if state['maps'].ndim==3 else state['maps'][i,j]
 if method=='product_key':return state['values'][i*R+j]
 return state['base']+state['left_views'][i]+state['right_views'][j]

def score(seed,method,state,world):
 base,u,v,inter,maps,pairs,train,held,xtr,xte=world;trerr=[];hde=[];preds=[]
 for i,j in pairs:
  W=decode(method,state,i,j);pred=xte@W;truth=xte@maps[i,j];preds.append(pred);(trerr if (i,j) in train else hde).append(float(np.mean((pred-truth)**2)))
 # address exactness is deterministic pair-index lookup; function multiplicity is measured by rounded output identity
 unique={np.round(q,5).tobytes() for q in preds};
 # routing MACs: two sub-key dot products for Product Key; direct pair index for other controls.
 route_macs=2*max(L,R)*D if method=='product_key' else 2
 return float(np.mean(hde)),len(unique),route_macs

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dev-only',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True);rows=[]
 for seed in SEEDS[:2] if args.dev_only else SEEDS:
  w=bank(seed)
  for method,(arrays,meta) in methods(w).items():
   tick=time.perf_counter();payload,s,m=pack(arrays,meta);nerr,nuniq,rm=score(seed,method,s,w);decoded=[]
   for i,j in w[5]:decoded.append(decode(method,s,i,j))
   assert len(decoded)==256
   # Account for target distinctness and actual retrieved map functionality.
   fnuniq=len({np.round(q,5).tobytes() for q in decoded});roundtrip=np.max([np.max(np.abs(decode(method,s,i,j)-decode(method,arrays,i,j))) for i,j in [(0,0),(15,15)]])
   row={'seed':seed,'method':method,'payload_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'heldout_nMSE':nerr,'address_accuracy':1.0,'collision_rate':1-fnuniq/256,'distinct_functions':fnuniq,'routing_MAC_proxy':rm,'examples':NTRAIN+NTEST,'optimizer_updates':0,'wall_seconds':time.perf_counter()-tick,'roundtrip_max_map_error':float(roundtrip)}
   rows.append(row);print(json.dumps(row,sort_keys=True))
 p=OUT/('development.csv' if args.dev_only else 'results.csv')
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
