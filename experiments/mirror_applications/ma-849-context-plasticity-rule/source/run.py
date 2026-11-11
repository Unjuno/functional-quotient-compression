import argparse,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=8; NTASK=24; STEPS=12

def tasks(seed):
 rng=np.random.default_rng(seed); out=[]
 for _ in range(NTASK):
  q,_=np.linalg.qr(rng.normal(size=(D,D)))
  eig=np.exp(rng.uniform(np.log(1.),np.log(30.),D))
  H=(q@np.diag(eig)@q.T).astype(np.float64); b=rng.normal(size=D)
  out.append((H,b))
 return out

def rollout(H,b,method,rate,beta=1.):
 x=np.zeros(D); xp=None; gp=None; mac=0
 for step in range(STEPS):
  g=H@(x-b); mac+=D*D
  eta=rate
  if method in ('native_context','mirror') and xp is not None:
   dx=x-xp; dg=g-gp; den=float(dx@dg)
   sec=float(dx@dx)/den if den>1e-15 else rate
   eta=float(np.clip(beta*sec,0.001,0.5)); mac+=3*D
  elif method=='native_context' or method=='mirror':
   eta=rate
  xp=x.copy(); gp=g.copy(); x=x-eta*g
 loss=.5*float((x-b)@H@(x-b))
 return loss,mac

def score(seed,method,rate,beta=1.):
 vals=[];mac=0
 for H,b in tasks(seed):
  l,m=rollout(H,b,method,rate,beta); vals.append(l);mac+=m
 return float(np.mean(vals)),mac

def serialize(method,rate,beta):
 path=ROOT/'source'/'params.npz'
 if method=='mirror': arrays={'base_rate':np.array([rate],dtype='<f4'),'mirror_code':np.array([beta],dtype='<f4')}
 elif method=='native_context': arrays={'base_rate':np.array([rate],dtype='<f4'),'native_gate':np.array([beta],dtype='<f4')}
 else: arrays={'learning_rate':np.array([rate],dtype='<f4')}
 np.savez(path,**arrays); size=path.stat().st_size+len(json.dumps({'D':D,'steps':STEPS,'method':method}).encode());path.unlink();return size

def tune():
 rates=np.linspace(.005,.12,48); fixed=min((score(17,'fixed',float(r))[0],float(r)) for r in rates)
 # Multiplicative coordinate selected on development; native/mirror use identical rule.
 candidates=np.linspace(.5,1.5,41); sec=min((score(17,'mirror',fixed[1],float(c))[0],float(c)) for c in candidates)
 return fixed[1],sec[1]

def run(seed,rate,beta):
 t=time.perf_counter(); rows=[]
 for method in ('fixed','native_context','mirror'):
  b=beta if method!='fixed' else 1.
  loss,mac=score(seed,method,rate,b)
  rows.append(dict(seed=seed,method=method,mean_final_excess_loss=loss,serialized_bytes=serialize(method,rate,b),updates=NTASK*STEPS,active_macs=mac,wall_time_s=time.perf_counter()-t,rate=rate,beta=b))
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--development',action='store_true');p.add_argument('--fresh',action='store_true');a=p.parse_args();r,b=tune();seeds=[17] if a.development else ([37,59] if a.fresh else [17,37,59]); print(json.dumps({'selected_rate':r,'selected_beta':b,'rows':[x for s in seeds for x in run(s,r,b)]},indent=2))
if __name__=='__main__':main()
