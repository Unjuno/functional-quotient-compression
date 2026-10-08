"""MA-881 synthetic routed translator-bank trajectory screen."""
from __future__ import annotations
import json,struct,time,hashlib
import numpy as np
D,E,ALIGNED,CTX,TOPK,STEPS=16,6,5,4,2,8
METHODS=('native_mot_full','independent_rank_r_delta','ordinary_shared_matrix_bank','mirror_shared_givens_views')

def rotvec(x,angles):
 y=np.array(x,copy=True)
 for i,t in enumerate(angles):
  a,b=x[2*i],x[2*i+1];c,s=np.cos(t),np.sin(t)
  y[2*i]=c*a-s*b;y[2*i+1]=s*a+c*b
 return y

def rotmat(angles):
 r=np.eye(D)
 for i,t in enumerate(angles):
  c,s=np.cos(t),np.sin(t);r[2*i:2*i+2,2*i:2*i+2]=((c,-s),(s,c))
 return r

def world(seed):
 rng=np.random.default_rng(seed);w0=rng.normal(size=(D,D))/np.sqrt(D)
 angles=rng.uniform(-0.45,0.45,size=(ALIGNED,D//2));weights=[rotmat(angles[e])@w0 for e in range(ALIGNED)]
 private=rng.normal(size=(D,D))/np.sqrt(D);weights.append(private);weights=np.stack(weights)
 router=rng.normal(size=(D,E))/np.sqrt(D);corr=rng.normal(size=(D,CTX))*0.25
 return w0,angles,weights,router,corr

def shared_factor(weights,q):
 w0=weights[:ALIGNED].mean(axis=0);delta=weights[:ALIGNED]-w0
 u,s,vt=np.linalg.svd(delta.reshape(ALIGNED,-1),full_matrices=False)
 basis=(vt[:q].reshape(q,D,D)*s[:q,None,None]).astype('<f4')
 coeff=u[:,:q].astype('<f4')
 return w0.astype('<f4'),basis,coeff

def independent_factor(weights,r):
 out=[]
 for e in range(ALIGNED):
  d=weights[e]-weights[:ALIGNED].mean(axis=0);u,s,vt=np.linalg.svd(d,full_matrices=False)
  out.append(((u[:,:r]*s[:r]).astype('<f4'),vt[:r].astype('<f4')))
 return out

def state(method,param,seed):
 w0,angles,weights,router,corr=world(seed);a={'router':router.astype('<f4'),'corr':corr.astype('<f4')};meta={'method':method,'seed':int(seed),'hidden':D,'experts':E,'topk':TOPK,'steps':STEPS}
 if method=='native_mot_full':a['weights']=weights.astype('<f4')
 elif method=='independent_rank_r_delta':
  r=int(param);a['w0']=weights[:ALIGNED].mean(axis=0).astype('<f4');a['private']=weights[-1].astype('<f4');meta['rank']=r
  for i,(u,v) in enumerate(independent_factor(weights,r)):a[f'u_{i}']=u;a[f'v_{i}']=v
 elif method=='ordinary_shared_matrix_bank':
  q=int(param);base,basis,coef=shared_factor(weights,q);a.update({'w0':base,'basis':basis,'coeff':coef,'private':weights[-1].astype('<f4')});meta['rank']=q
 elif method=='mirror_shared_givens_views':a.update({'w0':w0.astype('<f4'),'angles':angles.astype('<f4'),'private':weights[-1].astype('<f4')})
 else:raise ValueError(method)
 return meta,a

def serialize(meta,arrays):
 spec=[];chunks=[]
 for name in sorted(arrays):
  ar=np.ascontiguousarray(arrays[name],dtype='<f4');b=ar.tobytes();spec.append({'name':name,'shape':list(ar.shape),'nbytes':len(b)});chunks.append(b)
 h=json.dumps({'meta':meta,'arrays':spec},sort_keys=True,separators=(',',':')).encode()
 return b'MA881\x01'+struct.pack('<I',len(h))+h+b''.join(chunks)

def deserialize(b):
 assert b[:6]==b'MA881\x01';n=struct.unpack('<I',b[6:10])[0];h=json.loads(b[10:10+n]);pos=10+n;a={}
 for spec in h['arrays']:
  data=b[pos:pos+spec['nbytes']];pos+=spec['nbytes'];a[spec['name']]=np.frombuffer(data,dtype='<f4').copy().reshape(spec['shape'])
 assert pos==len(b)
 return h['meta'],a

def gates(x,router):
 logits=x@router/np.sqrt(D);idx=np.argpartition(logits,-TOPK)[-TOPK:];v=logits[idx];v=np.exp(v-v.max());v/=v.sum();return idx,v

def trajectory(method,meta,a,x0,contexts,teacher_weights=None):
 x=np.array(x0,dtype=np.float64);states=[];macs=0;rotation_flops=0;router_macs=0;corr_macs=0
 for ctx in contexts:
  idx,gw=gates(x,a['router']);router_macs+=D*E
  if method=='native_mot_full':
   trans=sum(float(g)*(a['weights'][int(e)]@x) for e,g in zip(idx,gw));macs+=len(idx)*D*D
  else:
   base=a['w0']@x;macs+=D*D;trans=np.zeros(D)
   for e,g in zip(idx,gw):
    e=int(e)
    if e==E-1:trans+=float(g)*(a['private']@x);macs+=D*D
    elif method=='mirror_shared_givens_views':
     trans+=float(g)*rotvec(base,a['angles'][e]);rotation_flops+=D*3
    elif method=='ordinary_shared_matrix_bank':
     y=base.copy()
     for j,b in enumerate(a['basis']):y+=a['coeff'][e,j]*(b@x);macs+=D*D
     trans+=float(g)*y
    elif method=='independent_rank_r_delta':
     r=a[f'u_{e}'].shape[1];d=a[f'u_{e}']@(a[f'v_{e}']@x);macs+=2*D*r;trans+=float(g)*(base+d)
  corrected=trans+a['corr']@ctx;corr_macs+=D*CTX;x=np.tanh(corrected);states.append(x.copy())
 return np.stack(states),{'translator_macs':macs,'router_macs':router_macs,'context_correction_macs':corr_macs,'givens_scalar_flops':rotation_flops}

def make_eval(seed,n=96):
 rng=np.random.default_rng(seed+71);x=rng.normal(size=(n,D));ctx=rng.normal(size=(n,STEPS,CTX));return x,ctx

def evaluate(seed,method,param):
 w0,angles,weights,router,corr=world(seed);meta,arr=state(method,param,seed);blob=serialize(meta,arr);meta,arr=deserialize(blob)
 x0,contexts=make_eval(seed);errs=[];norms=[];sums={k:0 for k in ('translator_macs','router_macs','context_correction_macs','givens_scalar_flops')};t0=time.perf_counter()
 for x,c in zip(x0,contexts):
  yref,_=trajectory('native_mot_full',{'method':'native_mot_full'},{'weights':weights.astype('<f4'),'router':router.astype('<f4'),'corr':corr.astype('<f4')},x,c)
  yhat,counts=trajectory(method,meta,arr,x,c)
  errs.extend(np.mean((yhat-yref)**2,axis=1).tolist());norms.extend(np.mean(yref**2,axis=1).tolist())
  for k in sums:sums[k]+=counts[k]
 wall=time.perf_counter()-t0
 return {'seed':seed,'method':method,'param':int(param) if param is not None else 0,'serialized_bytes':len(blob),'normalized_trajectory_mse':float(np.mean(errs)/max(float(np.mean(norms)),1e-30)),'translator_macs':sums['translator_macs'],'router_macs':sums['router_macs'],'context_correction_macs':sums['context_correction_macs'],'givens_scalar_flops':sums['givens_scalar_flops'],'wall_clock_s':wall,'trajectory_steps':len(x0)*STEPS,'payload_sha256':hashlib.sha256(blob).hexdigest()}

if __name__=='__main__':
 import csv,pathlib,sys
 root=pathlib.Path(__file__).resolve().parents[1];mode=sys.argv[1] if len(sys.argv)>1 else 'dev';dev_seeds=(88101,88102);fresh=(88111,88112,88113)
 if mode=='dev':
  rows=[]
  for seed in dev_seeds:
   for method in METHODS:
    params=([0] if method=='native_mot_full' else [1,2,4,8,16] if method=='independent_rank_r_delta' else [1,2,3,4,5] if method=='ordinary_shared_matrix_bank' else [0])
    for p in params:rows.append(evaluate(seed,method,p))
  with (root/'source/development.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
  chosen={}
  for method in ('independent_rank_r_delta','ordinary_shared_matrix_bank'):
   for p in sorted(set(r['param'] for r in rows if r['method']==method)):
    rr=[r for r in rows if r['method']==method and r['param']==p];m=sum(r['normalized_trajectory_mse'] for r in rr)/len(rr)
    if m<=1e-4:chosen[method]=p;break
  cfg={'selected_controls':chosen,'mirror_angles_per_aligned_expert':D//2,'fresh_seeds':list(fresh)}
  (root/'source/frozen_config.json').write_text(json.dumps(cfg,indent=2)+'\n');print(cfg)
 elif mode=='fresh':
  cfg=json.load(open(root/'source/frozen_config.json'));rows=[]
  for seed in fresh:
   for method in METHODS:
    params=([0] if method=='native_mot_full' else [cfg['selected_controls'][method]] if method in cfg['selected_controls'] else [0])
    for p in params:rows.append(evaluate(seed,method,p))
  with (root/'source/fresh.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
  for r in rows:print(r)
 else:raise SystemExit('usage: engine.py dev|fresh')
