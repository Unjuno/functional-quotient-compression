"""Synthetic held-out client personalization: phase basis vs pFedHN-like generator."""
import argparse,csv,hashlib,json,struct,time
from pathlib import Path
import numpy as np, torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];D,O,C,NTR,NTE=8,4,16,64,256
DEV=[34121,34122];FRESH=[34131,34132,34133];UPDATES=1000
METHODS=['shared_single','mirror_phase','generic_coefficients','pfedhn_generator','independent_client_fit']
torch.set_num_threads(1)
def world(seed):
 r=np.random.default_rng(seed); W0=(r.standard_normal((D,O),dtype=np.float32)*.25);A=(r.standard_normal((D,O),dtype=np.float32)*.18);B=(r.standard_normal((D,O),dtype=np.float32)*.18)
 ph_train=np.arange(C,dtype=np.float32)*np.float32(2*np.pi/C);ph_test=(np.arange(8,dtype=np.float32)+.25)*np.float32(2*np.pi/8)
 def examples(ph,n):
  xs=[];ys=[]
  for q in ph:
   x=r.standard_normal((n,D),dtype=np.float32);w=W0+np.cos(q)*A+np.sin(q)*B;y=x@w+r.standard_normal((n,O),dtype=np.float32)*.01;xs.append(x);ys.append(y)
  return np.stack(xs),np.stack(ys)
 xt,yt=examples(ph_train,NTR);xv,yv=examples(ph_test,NTE);xs,ys=examples(ph_test,NTR)
 return dict(W0=W0,A=A,B=B,ph_train=ph_train,ph_test=ph_test,xt=xt,yt=yt,xv=xv,yv=yv,xs=xs,ys=ys)
def tensors(a):return [torch.tensor(a[k]) for k in ('xt','yt','xv','yv','xs','ys','ph_train','ph_test')]
def fit_basis(a,kind):
 xt,yt,xv,yv,xs,ys,pt,pv=tensors(a); torch.manual_seed(901)
 w0=nn.Parameter(torch.zeros(D,O));u=nn.Parameter(torch.zeros(D,O));v=nn.Parameter(torch.zeros(D,O));opt=torch.optim.Adam([w0,u,v],lr=.035)
 zt=torch.stack([torch.cos(pt),torch.sin(pt)],-1);zv=torch.stack([torch.cos(pv),torch.sin(pv)],-1)
 start=time.perf_counter()
 for _ in range(UPDATES):
  opt.zero_grad(); co=torch.stack([torch.cos(pt),torch.sin(pt)],-1) if kind=='mirror' else zt
  mats=w0[None]+co[:,0,None,None]*u[None]+co[:,1,None,None]*v[None]
  pred=torch.einsum('cnd,cdo->cno',xt,mats);loss=((pred-yt)**2).mean();loss.backward();opt.step()
 sec=time.perf_counter()-start
 co=zv
 mats=(w0[None]+co[:,0,None,None]*u[None]+co[:,1,None,None]*v[None]).detach()
 pred=torch.einsum('cnd,cdo->cno',xv,mats);mse=float(((pred-yv)**2).mean())
 state=[w0.detach().numpy(),u.detach().numpy(),v.detach().numpy()]
 return mse,state,sec,float(loss.detach()),
def fit_shared(a):
 xt,yt,xv,yv,*_=tensors(a); w=nn.Parameter(torch.zeros(D,O));opt=torch.optim.Adam([w],lr=.035);t=time.perf_counter()
 for _ in range(UPDATES):opt.zero_grad();loss=((torch.einsum('cnd,do->cno',xt,w)-yt)**2).mean();loss.backward();opt.step()
 sec=time.perf_counter()-t;pred=torch.einsum('cnd,do->cno',xv,w);return float(((pred-yv)**2).mean()),[w.detach().numpy()],sec,float(loss.detach())
def fit_hyper(a):
 xt,yt,xv,yv,*rest=tensors(a);pt=rest[-2];pv=rest[-1];zt=torch.stack([torch.cos(pt),torch.sin(pt)],-1);zv=torch.stack([torch.cos(pv),torch.sin(pv)],-1)
 torch.manual_seed(902);net=nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,D*O));opt=torch.optim.Adam(net.parameters(),lr=.01);t=time.perf_counter()
 for _ in range(UPDATES):
  opt.zero_grad();mat=net(zt).reshape(C,D,O);pred=torch.einsum('cnd,cdo->cno',xt,mat);loss=((pred-yt)**2).mean();loss.backward();opt.step()
 sec=time.perf_counter()-t;mat=net(zv).reshape(8,D,O);pred=torch.einsum('cnd,cdo->cno',xv,mat);return float(((pred-yv)**2).mean()),[p.detach().numpy() for p in net.parameters()],sec,float(loss.detach())
def fit_independent(a):
 xv,yv,xs,ys=[torch.tensor(a[k]) for k in ('xv','yv','xs','ys')]; mats=[]
 for x,y in zip(xs,ys): mats.append(np.linalg.lstsq(x.numpy(),y.numpy(),rcond=None)[0].astype(np.float32))
 mm=torch.tensor(np.stack(mats));pred=torch.einsum('cnd,cdo->cno',xv,mm);return float(((pred-yv)**2).mean()),[mm.numpy()],0.,0.
def pack(method,state,client_codes):
 meta={'method':method,'shapes':[list(x.shape) for x in state+client_codes],'dtype':'float32','version':1};j=json.dumps(meta,sort_keys=True,separators=(',',':')).encode();return b'MA341\0'+struct.pack('<I',len(j))+j+b''.join(np.asarray(x,dtype=np.float32).tobytes() for x in state+client_codes)
def run(phase):
 rows=[]
 for seed in (DEV if phase=='development' else FRESH):
  a=world(seed);desc=np.stack([np.cos(a['ph_test']),np.sin(a['ph_test'])],-1).astype(np.float32);angle=a['ph_test'][:,None].astype(np.float32)
  results={}
  mse,state,sec,trainloss=fit_shared(a);results['shared_single']=(mse,state,sec,trainloss,desc)
  mse,state,sec,trainloss=fit_basis(a,'mirror');results['mirror_phase']=(mse,state,sec,trainloss,angle)
  mse,state,sec,trainloss=fit_basis(a,'generic');results['generic_coefficients']=(mse,state,sec,trainloss,desc)
  mse,state,sec,trainloss=fit_hyper(a);results['pfedhn_generator']=(mse,state,sec,trainloss,desc)
  mse,state,sec,trainloss=fit_independent(a);results['independent_client_fit']=(mse,state,sec,trainloss,np.zeros((8,0),np.float32))
  for method,(mse,state,sec,trainloss,codes) in results.items():
   blob=pack(method,state,[codes]);rows.append({'phase':phase,'world':seed,'method':method,'serialized_bytes':len(blob),'train_examples':C*NTR,'optimizer_updates':UPDATES if method!='independent_client_fit' else 0,'active_compute_proxy':UPDATES*C*NTR*D*O if method!='independent_client_fit' else 8*NTR*D*O,'wall_time_s':sec,'heldout_test_mse':mse,'train_mse':trainloss,'client_code_bytes':codes.nbytes,'payload_sha256':hashlib.sha256(blob).hexdigest()})
 path=ROOT/f'{phase.upper()}_RESULTS.csv'
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows),'seeds':DEV if phase=='development' else FRESH}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
