import argparse,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=16; NT=8; T=64; PREFIXES=(16,64)

def serialize(path,**a):
 np.savez(path,**a); n=path.stat().st_size
 with np.load(path) as z:
  assert set(z.files)==set(a)
  for k,v in a.items(): assert np.array_equal(z[k],v)
 path.unlink();return n+len(json.dumps({'D':D,'tasks':NT,'T':T,'dtype':'f32'}).encode())

def make(seed):
 rng=np.random.default_rng(seed); rates=rng.uniform(.1,.9,NT).astype('<f4'); streams=[]
 for t in range(NT):
  ks=rng.integers(0,D,T); vs=rng.normal(size=(T,D)).astype('<f4'); streams.append((ks,vs))
 return rates,streams

def rollout(rates,streams,method,fixed_rate=.5):
 errs=[]; update=0; query=0; reset=0
 W=np.zeros((D,D),np.float32)
 for t,(ks,vs) in enumerate(streams):
  if t>0: W.fill(0)
  ref=np.zeros_like(W)
  # Reuse one writable state object and clear it between task episodes.
  reset=max(reset,float(np.max(np.abs(W))))
  lam=float(fixed_rate if method=='fixed' else rates[t])
  for i,(ki,v) in enumerate(zip(ks,vs),1):
   k=np.eye(D,dtype=np.float32)[ki]
   W*=lam; W+=np.outer(v,k)
   ref*=float(rates[t]);ref+=np.outer(v,k)
   update+=2*D*D
   if i in PREFIXES:
    q=np.eye(D,dtype=np.float32)
    pred=q@W.T; target=q@ref.T
    errs.append(float(np.mean((pred-target)**2)/(np.mean(target**2)+1e-12)))
    query+=D*D*D
 return float(np.sqrt(np.mean(errs))),update,query,reset

def run(seed,fixed_rate):
 rates,streams=make(seed); path=ROOT/'source'/'payload.npz'; results={};t0=time.perf_counter()
 fullops=np.stack([np.eye(D,dtype='<f4')*r for r in rates]); states=np.zeros((NT,D,D),dtype='<f4')
 sizes={
  'fixed':serialize(path,shared_rate=np.array([fixed_rate],dtype='<f4'),states=states),
  'native_gate':serialize(path,task_rates=rates,states=states),
  'mirror_code':serialize(path,view_codes=rates,states=states),
  'independent_full_operator':serialize(path,operators=fullops,states=states)}
 for m in sizes:
  rate=fixed_rate if m=='fixed' else .5
  e,u,q,z=rollout(rates,streams,m,rate);results[m]=dict(nrmse=e,bytes=sizes[m],update_macs=u,query_macs=q,reset_max=z)
 return dict(seed=seed,results=results,examples=NT*T,updates=NT*T,writable_state_bank_bytes=NT*D*D*4,one_active_state_bytes=D*D*4,wall_time_s=time.perf_counter()-t0,task_rates=rates.tolist())

def tune():
 rates=np.linspace(.05,.95,91);return min((run(23,float(x))['results']['fixed']['nrmse'],float(x)) for x in rates)[1]

def main():
 a=argparse.ArgumentParser();a.add_argument('--development',action='store_true');a.add_argument('--fresh',action='store_true');o=a.parse_args();rate=tune();seeds=[23] if o.development else ([43,71] if o.fresh else [23,43,71]);print(json.dumps({'fixed_rate':rate,'rows':[run(s,rate) for s in seeds]},indent=2))
if __name__=='__main__':main()
