import argparse,csv,json,hashlib,math,time
from pathlib import Path
import numpy as np
D=8;N=512

def rotation(t):
 r=np.eye(D);c=np.cos(t);s=np.sin(t);r[:2,:2]=[[c,-s],[s,c]];return r

def hadamard():
 h=np.array([[1.]])
 while h.shape[0]<D:h=np.block([[h,h],[h,-h]])/np.sqrt(2.)
 return h

def quant(w):
 scale=np.max(np.abs(w))/7.;scale=max(scale,1e-12);q=np.clip(np.rint(w/scale),-7,7).astype(np.int8);return q,np.float32(scale)
def pack(q):
 n=q.astype(np.int16).ravel()&15
 if n.size%2:n=np.r_[n,0]
 return (n[0::2]|(n[1::2]<<4)).astype(np.uint8)
def unpack(p,shape):
 n=np.empty(p.size*2,dtype=np.int8);n[0::2]=(p&15).astype(np.int8);n[1::2]=((p>>4)&15).astype(np.int8);n[n>=8]-=16;return n[:np.prod(shape)].reshape(shape)
def serialize(path,w1,w2,method,angle=None):
 q1,s1=quant(w1);q2,s2=quant(w2);meta=np.array([method],np.int8);theta=np.array([0. if angle is None else angle],np.float32)
 np.savez_compressed(path,q1=pack(q1),q2=pack(q2),s=np.array([s1,s2],np.float32),method=meta,theta=theta)
 with np.load(path) as z:
  a=unpack(z['q1'],w1.shape).astype(np.float64)*float(z['s'][0]);b=unpack(z['q2'],w2.shape).astype(np.float64)*float(z['s'][1])
 return a,b,path.stat().st_size

def world(seed,split,root,angle=None):
 rng=np.random.default_rng(seed);w1=rng.normal(0,.25,(D,D));w2=rng.normal(0,.25,(D,D));w1[:,0]*=5.;w2[0,:]*=.2;x=rng.normal(size=(N,D));y=x@w1@w2
 out=Path(root)/split/str(seed);out.mkdir(parents=True,exist_ok=True);results=[];states={};decoded={}
 methods=[('identity',0.,None),('hadamard',1.,None),('mirror',2.,angle),('direct_angle',3.,angle)]
 for name,code,t in methods:
  r=np.eye(D) if name=='identity' else hadamard() if name=='hadamard' else rotation(float(t))
  a,b,nb=serialize(out/f'{name}.npz',w1@r,r.T@w2,code,t)
  decoded[name]=(a,b)
  pred=x@a@b;err=float(np.mean((pred-y)**2)/(np.mean(y*y)+1e-20));
  eq=float(np.max(np.abs((w1@r)@(r.T@w2)-w1@w2)))
  results.append({'seed':seed,'split':split,'method':name,'nrmse':math.sqrt(err),'payload_bytes':nb,'full_precision_gauge_max_abs_error':eq,'inference_macs':2*N*D*D,'calibration_evals':181 if split=='dev' and name=='mirror' else 0})
  states[name]=hashlib.sha256((out/f'{name}.npz').read_bytes()).hexdigest()
 np.savez_compressed(out/'full_precision.npz',W1=w1,W2=w2);fullbytes=(out/'full_precision.npz').stat().st_size
 with np.load(out/'full_precision.npz') as z:decoded['full_precision']=(z['W1'],z['W2'])
 results.append({'seed':seed,'split':split,'method':'full_precision','nrmse':0.,'payload_bytes':fullbytes,'full_precision_gauge_max_abs_error':0.,'inference_macs':2*N*D*D,'calibration_evals':0});states['full_precision']=hashlib.sha256((out/'full_precision.npz').read_bytes()).hexdigest()
 inference_ms={}
 for name,(aa,bb) in decoded.items():
  for _ in range(4):x@aa@bb
  st=time.perf_counter()
  for _ in range(40):x@aa@bb
  inference_ms[name]=(time.perf_counter()-st)*1000/40
  for row in results:
   if row['method']==name:row['inference_ms_512']=inference_ms[name]
 (out/'metrics.json').write_text(json.dumps({'seed':seed,'split':split,'angle':angle,'full_precision_bytes':fullbytes,'hashes':states,'gauge_equivalence_checked':True,'inference_ms_512':inference_ms},indent=2)+'\n')
 with (out/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
 return results

def main():
 a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--split',choices=['dev','fresh'],required=True);a.add_argument('--seeds',type=int,nargs='+',required=True);a.add_argument('--angle',type=float);z=a.parse_args();r=[]
 if z.split=='dev' and z.angle is None:
  # Select only on development inputs; fresh worlds never enter this grid search.
  rows=[];cal_start=time.perf_counter()
  for t in np.linspace(-np.pi/2,np.pi/2,181):
   vals=[]
   for seed in z.seeds:
    rng=np.random.default_rng(seed);w1=rng.normal(0,.25,(D,D));w2=rng.normal(0,.25,(D,D));w1[:,0]*=5.;w2[0,:]*=.2;x=rng.normal(size=(N,D));y=x@w1@w2;R=rotation(t);q1,s1=quant(w1@R);q2,s2=quant(R.T@w2);pred=x@(q1.astype(float)*s1)@(q2.astype(float)*s2);vals.append(np.mean((pred-y)**2)/(np.mean(y*y)+1e-20))
   rows.append((float(np.mean(vals)),float(t)))
  score,t=min(rows);calibration_seconds=time.perf_counter()-cal_start;Path(z.out,'dev').mkdir(parents=True,exist_ok=True);Path(z.out,'dev','ANGLE_SELECTION.json').write_text(json.dumps({'selected_angle':t,'mean_dev_nmse':math.sqrt(score),'grid_points':181,'seeds':z.seeds,'calibration_seconds':calibration_seconds},indent=2)+'\n');z.angle=t;print(json.dumps({'selected_angle':t,'mean_dev_nrmse':math.sqrt(score),'calibration_seconds':calibration_seconds}))
 for s in z.seeds:r+=world(s,z.split,z.out,z.angle)
 p=Path(z.out)/z.split/'RESULTS_CORE.csv';p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
if __name__=='__main__':main()
