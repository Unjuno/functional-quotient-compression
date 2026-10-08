#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"artifacts";SEEDS=(34901,34902,34911,34912,34913);D=32;N=12000;NSAMP=2

def pack(arrays,meta):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for n,a in sorted(arrays.items()):
   q=io.BytesIO();np.save(q,np.asarray(a),allow_pickle=False);i=zipfile.ZipInfo(n+'.npy',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,q.getvalue())
  i=zipfile.ZipInfo('metadata.json',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,json.dumps(meta,sort_keys=True,separators=(',',':')).encode())
 data=b.getvalue()
 with zipfile.ZipFile(io.BytesIO(data)) as z:
  aa={n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')};mm=json.loads(z.read('metadata.json'))
 return data,aa,mm

def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-40,40)))
def ece(p,y,bins=15):
 edges=np.linspace(0,1,bins+1);val=0.
 for lo,hi in zip(edges[:-1],edges[1:]):
  mask=(p>=lo)&(p<(hi if hi<1 else hi+1e-8))
  if np.any(mask):val+=float(mask.mean())*abs(float(p[mask].mean())-float(y[mask].mean()))
 return val

def world(seed):
 rng=np.random.default_rng(seed);w=rng.normal(size=D).astype(np.float32)*.12;j=int(rng.integers(0,D));w[j]=.7;delta=1.8;mode=np.array([-1,1],np.int8);prob=np.array([.5,.5],np.float32)
 x=rng.normal(size=(N,D)).astype(np.float32);w0=w.copy();wminus=w.copy();wplus=w.copy();wminus[j]-=delta;wplus[j]+=delta
 ptrue=.5*sigmoid(x@wminus)+.5*sigmoid(x@wplus);y=rng.binomial(1,ptrue).astype(np.float32)
 return rng,w,j,delta,mode,prob,x,y,ptrue,wminus,wplus

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dev-only',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True);rows=[]
 for seed in SEEDS[:2] if args.dev_only else SEEDS:
  rng,w,j,d,mode,prob,x,y,ptrue,wminus,wplus=world(seed);scale=np.array([d/w[j]],np.float32);idx=np.array([j],np.uint16)
  cases={
   'deterministic_mean':({'weight':w},{'kind':'mean'}),
   'independent_two_model_posterior':({'weights':np.stack([wminus,wplus]),'probabilities':prob},{'kind':'independent_two_mode'}),
   'rank1_bnn_multiplicative_posterior':({'mean':w,'factor_index':idx,'factor_scale':scale,'factor_modes':mode,'probabilities':prob},{'kind':'rank1_multiplicative_posterior','samples':NSAMP}),
   'mirror_scalar_posterior':({'mean':w,'direction_index':idx,'direction_value':np.array([d],np.float32),'view_codes':mode,'probabilities':prob},{'kind':'sparse_scalar_posterior','samples':NSAMP}),
   'direct_sparse_scalar_posterior':({'mean':w,'direction_index':idx,'direction_value':np.array([d],np.float32),'view_codes':mode,'probabilities':prob},{'kind':'sparse_scalar_posterior','samples':NSAMP})}
  for name,(arrays,meta) in cases.items():
   tick=time.perf_counter();payload,ld,mm=pack(arrays,meta)
   if name=='deterministic_mean':p=sigmoid(x@ld['weight']);ns=1
   elif name=='independent_two_model_posterior':p=np.sum(ld['probabilities'][None,:]*sigmoid(x@ld['weights'].T),axis=1);ns=2
   elif name=='rank1_bnn_multiplicative_posterior':
    weights=[]
    for eps in ld['factor_modes']:
     a=ld['mean'].copy();jj=int(ld['factor_index'][0]);a[jj]+=float(eps)*ld['mean'][jj]*float(ld['factor_scale'][0]);weights.append(a)
    p=np.sum(ld['probabilities'][None,:]*sigmoid(x@np.stack(weights).T),axis=1);ns=NSAMP
   else:
    weights=[]
    for m in ld['view_codes']:
     a=ld['mean'].copy();a[int(ld['direction_index'][0])]+=float(m)*float(ld['direction_value'][0]);weights.append(a)
    p=np.sum(ld['probabilities'][None,:]*sigmoid(x@np.stack(weights).T),axis=1);ns=NSAMP
   p=np.clip(p,1e-7,1-1e-7);nll=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)));brier=float(np.mean((p-y)**2));cal=float(np.mean((p-ptrue)**2));wall=time.perf_counter()-tick
   rows.append({'seed':seed,'method':name,'payload_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'predictive_nll':nll,'brier':brier,'ece_15bin':ece(p,y),'probability_calibration_mse':cal,'posterior_modes':2 if ns>1 else 1,'posterior_samples_per_input':ns,'examples_evaluated':N,'optimizer_updates':0,'active_MACs':int(N*D*ns),'wall_seconds_serialize_predict':wall})
 out=OUT/('development.csv' if args.dev_only else 'results.csv')
 with out.open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');wri.writeheader();wri.writerows(rows)
 for r in rows:print(json.dumps(r,sort_keys=True))
if __name__=='__main__':main()
