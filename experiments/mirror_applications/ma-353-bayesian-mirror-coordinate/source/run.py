#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';SEEDS=(35301,35302,35311,35312,35313);D=16;N=10000
GH_X=np.array([-2.020182870456086,-0.958572464613819,0.,0.958572464613819,2.020182870456086],np.float32)*np.sqrt(2)
GH_W=np.array([0.0199532420590459,0.393619323152241,0.945308720482942,0.393619323152241,0.0199532420590459],np.float32)/np.sqrt(np.pi)
MODES=np.array([-1.,0.,1.],np.float32);PROBS=np.ones(3,np.float32)/3

def pack(arrays,meta):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for name,a in sorted(arrays.items()):
   q=io.BytesIO();np.save(q,np.asarray(a),allow_pickle=False);i=zipfile.ZipInfo(name+'.npy',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,q.getvalue())
  i=zipfile.ZipInfo('metadata.json',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,json.dumps(meta,sort_keys=True,separators=(',',':')).encode())
 data=b.getvalue()
 with zipfile.ZipFile(io.BytesIO(data)) as z:
  state={n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')};metadata=json.loads(z.read('metadata.json'))
 return data,state,metadata

def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-40,40)))
def ece(p,y,bins=15):
 out=0.;edges=np.linspace(0,1,bins+1)
 for lo,hi in zip(edges[:-1],edges[1:]):
  m=(p>=lo)&(p<(hi if hi<1 else hi+1e-8))
  if m.any():out+=float(m.mean())*abs(float(p[m].mean())-float(y[m].mean()))
 return out
def world(seed):
 rng=np.random.default_rng(seed);w=rng.normal(0,.08,D).astype('f4');j=int(rng.integers(D));w[j]=.55;direction=np.float32(.9);modes=MODES.copy()
 x=rng.normal(size=(N,D)).astype('f4');shift=rng.normal(0,1.5,size=(N,D)).astype('f4');xood=x+shift
 weights=np.stack([w+float(m)*direction*np.eye(1,D,j,dtype='f4').reshape(D) for m in modes]);ptrue=np.mean(sigmoid(x@weights.T),axis=1);y=rng.binomial(1,ptrue).astype('f4')
 pood=np.mean(sigmoid(xood@weights.T),axis=1)
 return w,j,direction,x,xood,y,ptrue,pood,weights

def posterior(method,w,j,direction):
 if method=='deterministic_mean':return {'weight':w},{'kind':'deterministic'}
 if method=='independent_three_model':return {'weights':np.stack([w-direction*np.eye(1,D,j,dtype='f4').reshape(D),w,w+direction*np.eye(1,D,j,dtype='f4').reshape(D)]),'probabilities':PROBS},{'kind':'independent_three_mode'}
 if method=='rank1_bnn_gaussian':return {'mean':w,'factor_index':np.array([j],np.uint16),'factor_scale':np.array([direction/w[j]],'f4'),'factor_mu':np.array([0.],'f4'),'factor_sigma':np.array([1.],'f4'),'gh_x':GH_X,'gh_w':GH_W},{'kind':'rank1_gaussian_factor','quadrature_nodes':5}
 if method in ('mirror_gaussian_coordinate','direct_gaussian_scalar'):
  return {'mean':w,'direction_index':np.array([j],np.uint16),'direction_value':np.array([direction],'f4'),'coordinate_mu':np.array([0.],'f4'),'coordinate_sigma':np.array([1.],'f4'),'gh_x':GH_X,'gh_w':GH_W},{'kind':'gaussian_scalar_coordinate','quadrature_nodes':5}
 raise ValueError(method)

def predict(method,state,x):
 if method=='deterministic_mean':return sigmoid(x@state['weight'])
 if method=='independent_three_model':return np.sum(state['probabilities'][None,:]*sigmoid(x@state['weights'].T),axis=1)
 if method=='rank1_bnn_gaussian':
  j=int(state['factor_index'][0]);base=state['mean'];fac=float(state['factor_scale'][0]);mu=float(state['factor_mu'][0]);sig=float(state['factor_sigma'][0]);p=[]
  for q in state['gh_x']:
   w=base.copy();w[j]+=float((mu+sig*q)*fac*base[j]);p.append(sigmoid(x@w))
  return np.sum(state['gh_w'][None,:]*np.stack(p,axis=1),axis=1)
 j=int(state['direction_index'][0]);base=state['mean'];direction=float(state['direction_value'][0]);mu=float(state['coordinate_mu'][0]);sig=float(state['coordinate_sigma'][0]);p=[]
 for q in state['gh_x']:
  w=base.copy();w[j]+=(mu+sig*float(q))*direction;p.append(sigmoid(x@w))
 return np.sum(state['gh_w'][None,:]*np.stack(p,axis=1),axis=1)

def metrics(p,y,ptrue):
 p=np.clip(p,1e-7,1-1e-7)
 return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),float(np.mean((p-y)**2)),ece(p,y),float(np.mean((p-ptrue)**2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dev-only',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True);rows=[]
 for seed in SEEDS[:2] if args.dev_only else SEEDS:
  w,j,d,x,xood,y,ptrue,pood,weights=world(seed)
  for method in ['deterministic_mean','independent_three_model','rank1_bnn_gaussian','mirror_gaussian_coordinate','direct_gaussian_scalar']:
   start=time.perf_counter();arrays,meta=posterior(method,w,j,d);payload,state,loadedmeta=pack(arrays,meta)
   pin=predict(method,state,x);po=predict(method,state,xood);q=metrics(pin,y,ptrue);ood_nll=metrics(po,(pood>=.5).astype('f4'),pood)[0]
   replay=predict(method,state,x);assert np.array_equal(pin,replay)
   modes=3 if method in ('independent_three_model',) else (5 if method not in ('deterministic_mean',) else 1);macs=N*D*modes
   row={'seed':seed,'method':method,'payload_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'predictive_nll':q[0],'brier':q[1],'ece_15bin':q[2],'calibration_mse':q[3],'ood_nll':ood_nll,'components_per_input':modes,'examples':N,'optimizer_updates':0,'active_MACs':macs,'wall_seconds_serialize_predict':time.perf_counter()-start}
   rows.append(row);print(json.dumps(row,sort_keys=True))
 out=OUT/('development.csv' if args.dev_only else 'results.csv')
 with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
