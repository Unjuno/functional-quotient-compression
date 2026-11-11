"""Common support/query adaptation screen; hidden task teachers are evaluator-only."""
import argparse,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DI=8; DO=4; R=4; NTRAIN=128; NDEV=16; NFRESH=16; NS=32; NQ=128; CAVIA_STEPS=32; SIGMA=.05; EPS=1e-8

def teacher_family():
 rng=np.random.default_rng(5007)
 base=(rng.normal(size=(DO,DI))*.2).astype(np.float64)
 u,_=np.linalg.qr(rng.normal(size=(DO,2)));v,_=np.linalg.qr(rng.normal(size=(DI,2)))
 return base,u,v

def episodes(seed,n,teacher,n_support=NS):
 base,u,v=teacher;rng=np.random.default_rng(seed); out=[]
 for _ in range(n):
  c=rng.normal(size=(2,2))*.35; residual=rng.normal(size=(DO,DI))*SIGMA
  A=base+u@c@v.T+residual
  X=rng.normal(size=(n_support,DI));Y=X@A.T
  Xq=rng.normal(size=(NQ,DI));Yq=Xq@A.T
  out.append((X,Y,Xq,Yq))
 return out

def train_shared():
 train_t0=time.perf_counter();teacher=teacher_family(); train=episodes(101,NTRAIN,teacher,n_support=64); mats=[]
 for X,Y,_,_ in train:
  mats.append(Y.T@X@np.linalg.inv(X.T@X+1e-8*np.eye(DI)))
 mats=np.stack(mats); base=mats.mean(axis=0);delta=(mats-base).reshape(NTRAIN,-1)
 _,_,vt=np.linalg.svd(delta,full_matrices=False);basis=vt[:R].reshape(R,DO,DI)
 train_macs=NTRAIN*(64*DI*DI+64*DO*DI+DI**3+DO*DI*DI)+2*NTRAIN*(DO*DI)**2
 return base,basis,teacher,NTRAIN*64,train_macs,time.perf_counter()-train_t0

def predict(base,basis,state,x,method):
 if method in ('cavia','mirror','native_lowrank'):
  A=base+np.einsum('r,roi->oi',state,basis)
 else:A=base+state
 return x@A.T

def adapt_one(ep,base,basis,method,lr,initial_state=None):
 X,Y,Xq,Yq=ep;start=time.perf_counter();state=initial_state if initial_state is not None else np.zeros(R if method in ('cavia','mirror','native_lowrank') else (DO,DI));updates=0;mac=0
 if method=='zero':pass
 elif method=='hebb':
  residual=Y-X@base.T;state=(residual.T@X)/NS;updates=NS;mac=2*NS*DO*DI
 elif method=='delta':
  for x,y in zip(X,Y):
   e=y-(base+state)@x;state+=lr*np.outer(e,x)/(x@x+EPS);updates+=1;mac+=2*DO*DI+DI
 elif method in ('mirror','native_lowrank'):
  for x,y in zip(X,Y):
   phi=np.einsum('roi,i->ro',basis,x);pred=base@x+state@phi;e=y-pred
   grad=phi@e;state+=lr*grad/(np.sum(phi*phi)+EPS);updates+=1;mac+=DO*DI+R*DO*DI+3*R*DO
 elif method=='cavia':
  for _ in range(CAVIA_STEPS):
   pred=X@base.T+np.einsum('nri,r->ni',np.einsum('roi,ni->nro',basis,X),state)
   err=pred-Y;phi=np.einsum('roi,ni->nro',basis,X)
   grad=np.einsum('nro,no->r',phi,err)/(NS*DO);state-=lr*grad;updates+=1
   mac+=NS*DO*DI+NS*R*DO*DI+2*NS*R*DO
 elif method=='ridge':
  residual=Y-X@base.T;state=residual.T@X@np.linalg.inv(X.T@X+1e-4*np.eye(DI));updates=1;mac=NS*DO*DI+NS*DI*DI+NS*DO*DI+DI**3+DO*DI*DI
 else:raise ValueError(method)
 predq=predict(base,basis,state,Xq,method);err=float(np.mean((predq-Yq)**2)/(np.mean(Yq**2)+EPS));qmac=NQ*DO*DI+(R*DO*DI if method in ('cavia','mirror','native_lowrank') else 0)
 return state,err,updates,mac,qmac,time.perf_counter()-start

def serialize(method,base,basis,states):
 path=ROOT/'source'/'payload.npz'
 arr={'shared_mean':base.astype('<f4')}
 if method in ('cavia','mirror','native_lowrank'):arr.update(decoder=basis.astype('<f4'),task_states=np.stack(states).astype('<f4'))
 elif method=='zero':arr['empty_state']=np.zeros((1,),dtype='<f4')
 else:arr['task_states']=np.stack(states).astype('<f4')
 np.savez(path,**arr);size=path.stat().st_size
 with np.load(path) as z:
  assert set(z.files)==set(arr)
  for k,v in arr.items():assert np.array_equal(z[k],v)
 path.unlink();return size+len(json.dumps({'state_schema':'linear-task-bank-v1','D_in':DI,'D_out':DO,'tasks':len(states)}).encode())

def score(seed,lr_delta,lr_code,lr_ctx):
 base,basis,teacher,train_examples,train_macs,train_wall=train_shared();eps=episodes(seed,NDEV if seed==29 else NFRESH,teacher)
 methods=['zero','hebb','delta','cavia','mirror','native_lowrank','ridge'];result={}
 for method in methods:
  states=[];errs=[];ups=mac=qmac=0;wall=0.
  state_obj=None
  for ep in eps:
   lr=lr_ctx if method=='cavia' else lr_delta if method=='delta' else lr_code
   if state_obj is None: state_obj=np.zeros(R if method in ('cavia','mirror','native_lowrank') else (DO,DI))
   else: state_obj.fill(0)
   st,e,u,m,q,t=adapt_one(ep,base,basis,method,lr,state_obj);state_obj=st;states.append(st.copy());errs.append(e);ups+=u;mac+=m;qmac+=q;wall+=t
  result[method]=dict(nrmse=float(np.sqrt(np.mean(errs))),bytes=serialize(method,base,basis,states),adaptation_updates=ups,online_macs=mac,query_macs=qmac,wall_time_s=wall,state_bytes=0 if method=='zero' else int(np.stack(states).nbytes),snapshot_state_bytes=0 if method=='zero' else int(np.stack(states).astype('<f4').nbytes),persistent_shared_bytes=int(base.astype('<f4').nbytes+(basis.astype('<f4').nbytes if method in ('cavia','mirror','native_lowrank') else 0)))
 return {'methods':result,'shared_training_examples':train_examples,'shared_training_macs':train_macs,'shared_training_wall_time_s':train_wall}

def tune():
 best={}
 for method,grid in [('delta',[.8,1.0,1.2]),('code',[.21,.23,.25,.27,.29]),('ctx',[1.6,2.0,3.2,4.8])]:
  candidates=[]
  for rate in grid:
   rates={'delta':.2,'code':.2,'ctx':.8};rates[method]=rate
   res=score(29,rates['delta'],rates['code'],rates['ctx'])
   key={'delta':'delta','code':'mirror','ctx':'cavia'}[method];candidates.append((res['methods'][key]['nrmse'],rate))
  best[method]=min(candidates)[1]
 return best

def main():
 p=argparse.ArgumentParser();p.add_argument('--development',action='store_true');p.add_argument('--fresh',action='store_true');a=p.parse_args();params=tune();seeds=[29] if a.development else ([73,89] if a.fresh else [29,73,89]);print(json.dumps({'selected_rates':params,'training_examples_for_shared_basis':NTRAIN*64,'results':{s:score(s,params['delta'],params['code'],params['ctx']) for s in seeds}},indent=2))
if __name__=='__main__':main()
