#!/usr/bin/env python3
"""MA-600 hash-compressed adapter bank mechanism screen."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np, sklearn, torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

TASKS,INP,OUT,RANK,BUCKETS,UPDATES,BATCH=8,64,16,2,512,1200,128
METHODS=('independent_dense','independent_lora','independent_hash','shared_salted_hash','mirror_givens_hash','vera_shared_basis','generic_shared_basis')
DATA_HASH='faf2d98f61250c1cdc0b114323133d3c58da04f5c5d28bb78e4bde3c936a75b1'

def data():
 d=load_digits();raw=np.ascontiguousarray(d.data.astype(np.float32));y=np.ascontiguousarray(d.target.astype(np.int64));return raw/16.,hashlib.sha256(raw.tobytes()+y.tobytes()).hexdigest()
def hash_map(seed):
 r=np.random.default_rng(int(seed)&0xffffffff);return r.integers(0,BUCKETS,(INP,OUT),dtype=np.int64),r.choice(np.array([-1.,1.],np.float32),(INP,OUT))
def teacher(seed):
 r=np.random.default_rng(seed+770031);aa=[];bb=[]
 for _ in range(TASKS):
  aa.append(r.normal(0,0.08,(INP,RANK)).astype(np.float32));bb.append(r.normal(0,0.10,(RANK,OUT)).astype(np.float32))
 return np.stack(aa),np.stack(bb)
def givens(x,a):
 e,o=x[...,0::2],x[...,1::2];c,s=torch.cos(a),torch.sin(a);z=torch.empty_like(x);z[...,0::2]=c*e-s*o;z[...,1::2]=s*e+c*o;return z
def params(method,seed):
 torch.manual_seed(seed*101+600)
 p={}
 if method=='independent_dense':p['w']=torch.nn.Parameter(torch.randn(TASKS,INP,OUT)*.01)
 elif method=='independent_lora':p['a']=torch.nn.Parameter(torch.randn(TASKS,INP,RANK)*.01);p['b']=torch.nn.Parameter(torch.randn(TASKS,RANK,OUT)*.01)
 elif method in ('independent_hash','shared_salted_hash','mirror_givens_hash'):
  nt=TASKS if method=='independent_hash' else 1;p['theta']=torch.nn.Parameter(torch.randn(nt,BUCKETS)*.02)
  if method=='mirror_givens_hash':p['angles']=torch.nn.Parameter(torch.zeros(TASKS,INP//2))
 elif method=='vera_shared_basis':
  rr=np.random.default_rng(seed+991);p['fixed_a']=torch.as_tensor(rr.normal(0,.08,(INP,8)).astype(np.float32));p['fixed_b']=torch.as_tensor(rr.normal(0,.08,(8,OUT)).astype(np.float32));p['scale']=torch.nn.Parameter(torch.zeros(TASKS,8))
 elif method=='generic_shared_basis':p['basis']=torch.nn.Parameter(torch.randn(INP*OUT,8)*.01);p['code']=torch.nn.Parameter(torch.zeros(TASKS,8))
 return p
def state(method,p,seed,salts):
 q={k:v.detach().cpu().numpy().astype(np.float16) for k,v in p.items()};q.update({'method':np.frombuffer(method.encode(),np.uint8),'world_seed':np.asarray([seed],np.uint32),'salts':np.asarray(salts,np.uint32),'shape':np.asarray([TASKS,INP,OUT],np.int32),'schema':np.asarray([600,1],np.int32)});return q
def load(path):
 with np.load(path,allow_pickle=False) as f:q={k:f[k].copy() for k in f.files}
 m=bytes(q.pop('method').tolist()).decode();seed=int(q.pop('world_seed')[0]);salts=q.pop('salts');q.pop('shape');q.pop('schema');p={k:torch.as_tensor(v,dtype=torch.float32) for k,v in q.items()};maps=[hash_map(int(salts[t] if len(salts)==TASKS else salts[0])) for t in range(TASKS)];return m,seed,p,maps
def matrices(m,p,maps):
 if m=='independent_dense':return p['w']
 if m=='independent_lora':return torch.einsum('tir,tro->tio',p['a'],p['b'])
 if m in ('independent_hash','shared_salted_hash','mirror_givens_hash'):
  ws=[]
  for t,(idx,sg) in enumerate(maps):
   table=t if m=='independent_hash' else 0;w=(p['theta'][table][torch.as_tensor(idx)]*torch.as_tensor(sg)).reshape(INP,OUT)
   if m=='mirror_givens_hash':
    # W_eff satisfies x G_t W == x W_eff.
    eye=torch.eye(INP);w=givens(eye,p['angles'][t])@w
   ws.append(w)
  return torch.stack(ws)
 if m=='vera_shared_basis':return torch.stack([p['fixed_a']@torch.diag(p['scale'][t])@p['fixed_b'] for t in range(TASKS)])
 if m=='generic_shared_basis':return torch.stack([p['basis']@p['code'][t] for t in range(TASKS)]).reshape(TASKS,INP,OUT)
 raise ValueError(m)
def fit(method,x,y,seed,mi):
 p=params(method,seed+mi*100);salts=[(seed*7919+600+t*104729)&0xffffffff for t in range(TASKS)] if method in ('independent_hash','shared_salted_hash') else [(seed*7919+600)&0xffffffff]
 maps=[hash_map(salts[t] if len(salts)==TASKS else salts[0]) for t in range(TASKS)]
 # Frozen VeRA matrices and deterministic lookup indices are state, not trainable optimizer values.
 opt=torch.optim.AdamW([v for v in p.values() if isinstance(v,torch.nn.Parameter)],lr=.01,weight_decay=0.)
 rng=np.random.default_rng(seed*1000);xt=torch.tensor(x);yt=torch.tensor(y);start=time.perf_counter();loss0=None
 for st in range(UPDATES):
  ids=rng.integers(0,len(xt),BATCH);xb=xt[ids];yb=yt[ids];w=matrices(method,p,maps);pred=torch.einsum('bi,tio->bto',xb,w);loss=((pred-yb)**2).mean()
  if st==0:loss0=float(loss.detach())
  opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return p,salts,maps,time.perf_counter()-start,loss0,float(loss.detach())
def evaluate(path,x,y,aa,bb):
 m,seed,p,maps=load(path);xx=torch.tensor(x);targ=torch.einsum('ni,tir,tro->nto',xx,torch.tensor(aa),torch.tensor(bb));w=matrices(m,p,maps);start=time.perf_counter();pred=torch.einsum('ni,tio->nto',xx,w);infer=time.perf_counter()-start;err=(pred-targ)**2;den=(targ**2).mean((0,2)).clamp_min(1e-10);nmse=(err.mean((0,2))/den).numpy();rmse=torch.sqrt(err.mean((0,2))).numpy();return {'per_task_nrmse':np.sqrt(nmse).tolist(),'aggregate_nrmse':float(np.sqrt(err.mean().item()/((targ**2).mean().item()+1e-12))),'aggregate_rmse':float(torch.sqrt(err.mean()).item()),'per_task_rmse':rmse.tolist(),'inference_seconds':infer,'task_examples_per_second':len(x)*TASKS/infer}
def run(seed,split,out):
 torch.set_num_threads(1);x,dig=data();assert dig==DATA_HASH;xt,xv,_,_=train_test_split(x,np.zeros(len(x)),test_size=.25,random_state=seed,stratify=load_digits().target);aa,bb=teacher(seed);yt=np.einsum('ni,tir,tro->nto',xt,aa,bb).astype(np.float32);yv=np.einsum('ni,tir,tro->nto',xv,aa,bb).astype(np.float32);out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];all0=time.perf_counter()
 for mi,m in enumerate(METHODS):
  p,salts,maps,wall,l0,l1=fit(m,xt,yt,seed,mi);path=out/(m+'.npz');np.savez(path,**state(m,p,seed,salts));ev=evaluate(path,xv,yv,aa,bb);arr=matrices(m,load(path)[2],load(path)[3]);mac=TASKS*INP*OUT
  rows.append({'method':m,'bytes':path.stat().st_size,'payload_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'training_seconds':wall,'updates':UPDATES,'examples_seen':UPDATES*BATCH,'adapter_macs_per_task_example':INP*OUT,'initial_final_train_mse':[l0,l1],**ev})
 report={'experiment_id':'MA-600','world_seed':seed,'split':split,'dataset':'sklearn.load_digits','sklearn_version':sklearn.__version__,'dataset_hash':dig,'teacher':'independent seeded rank-2 matrices for eight tasks','methods':rows,'common_compute':{'device':'cpu','threads':1,'updates_per_method':UPDATES,'total_wall_seconds':time.perf_counter()-all0}}
 (out/'metrics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
def main():
 a=argparse.ArgumentParser();a.add_argument('--seed',type=int,required=True);a.add_argument('--split',choices=['dev','fresh'],required=True);a.add_argument('--out',type=Path,required=True);x=a.parse_args();run(x.seed,x.split,x.out)
if __name__=='__main__':main()
