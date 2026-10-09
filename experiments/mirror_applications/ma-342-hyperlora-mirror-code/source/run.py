"""MA-342 synthetic low-rank client adaptation with gauge-invariant delta metrics."""
import argparse,csv,hashlib,json,struct,time
from pathlib import Path
import numpy as np,torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];D=O=8;R=2;CT=16;NTR=64;NTE=256;UPDATES=1000;DEV=[34221,34222];FRESH=[34231,34232,34233]
torch.set_num_threads(1)
def world(seed):
 r=np.random.default_rng(seed);base=(r.standard_normal((D,O),dtype=np.float32)*.2).astype(np.float32);u=(r.standard_normal((D,R),dtype=np.float32)*.15).astype(np.float32);v=(r.standard_normal((O,R),dtype=np.float32)*.15).astype(np.float32)
 phtr=np.arange(CT,dtype=np.float32)*np.float32(2*np.pi/CT);phte=(np.arange(8,dtype=np.float32)+.25)*np.float32(2*np.pi/8)
 def make(ph,n):
  x=r.standard_normal((len(ph),n,D),dtype=np.float32);ys=[]
  for i,q in enumerate(ph):
   delta=u@np.diag([np.cos(q),np.sin(q)])@v.T;ys.append(x[i]@(base+delta)+r.standard_normal((n,O),dtype=np.float32)*.01)
  return x,np.stack(ys).astype(np.float32)
 xt,yt=make(phtr,NTR);xe,ye=make(phte,NTE);xs,ys=make(phte,NTR)
 return dict(base=base,u=u,v=v,phtr=phtr,phte=phte,xt=xt,yt=yt,xe=xe,ye=ye,xs=xs,ys=ys)
def t(a,k):return torch.tensor(a[k])
def basis(a,mirror):
 xt,yt,xe,ye=t(a,'xt'),t(a,'yt'),t(a,'xe'),t(a,'ye');phtr=t(a,'phtr');phte=t(a,'phte');base=t(a,'base');torch.manual_seed(810)
 u=nn.Parameter(torch.randn(D,R)*.1);v=nn.Parameter(torch.randn(O,R)*.1);opt=torch.optim.Adam([u,v],lr=.025);st=time.perf_counter()
 for _ in range(UPDATES):
  opt.zero_grad();z=torch.stack([torch.cos(phtr),torch.sin(phtr)],-1);delta=torch.einsum('dr,cr,or->cdo',u,z,v);pred=torch.einsum('cnd,do->cno',xt,base)+torch.einsum('cnd,cdo->cno',xt,delta);loss=((pred-yt)**2).mean();loss.backward();opt.step()
 sec=time.perf_counter()-st;z=torch.stack([torch.cos(phte),torch.sin(phte)],-1);delta=torch.einsum('dr,cr,or->cdo',u,z,v);pred=torch.einsum('cnd,do->cno',xe,base)+torch.einsum('cnd,cdo->cno',xe,delta);mse=float(((pred-ye)**2).mean());return mse,[u.detach().numpy(),v.detach().numpy()],sec,float(loss.detach())
def hyper(a):
 xt,yt,xe,ye=t(a,'xt'),t(a,'yt'),t(a,'xe'),t(a,'ye');phtr=t(a,'phtr');phte=t(a,'phte');base=t(a,'base');zt=torch.stack([torch.cos(phtr),torch.sin(phtr)],-1);ze=torch.stack([torch.cos(phte),torch.sin(phte)],-1);torch.manual_seed(811)
 net=nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,D*R+R*O));opt=torch.optim.Adam(net.parameters(),lr=.01);st=time.perf_counter()
 def predict(x,z):
  raw=net(z);aa=raw[:,:D*R].reshape(-1,D,R);bb=raw[:,D*R:].reshape(-1,R,O);delta=aa@bb;return torch.einsum('cnd,do->cno',x,base)+torch.einsum('cnd,cdo->cno',x,delta)
 for _ in range(UPDATES):opt.zero_grad();loss=((predict(xt,zt)-yt)**2).mean();loss.backward();opt.step()
 sec=time.perf_counter()-st;pred=predict(xe,ze);return float(((pred-ye)**2).mean()),[p.detach().numpy() for p in net.parameters()],sec,float(loss.detach())
def independent(a):
 base=a['base']; mats=[]
 for x,y in zip(a['xs'],a['ys']):
  w=np.linalg.lstsq(x,y,rcond=None)[0];u,s,vh=np.linalg.svd(w-base,full_matrices=False);delta=(u[:,:R]*s[:R])@vh[:R];mats.append(delta.astype(np.float32))
 pred=np.stack([x@(base+d) for x,d in zip(a['xe'],mats)]);mse=float(np.mean((pred-a['ye'])**2));return mse,[np.stack(mats)],0.,0.
def shared(a):
 return float(np.mean((a['xe']@a['base']-a['ye'])**2)),[],0.,0.
def pack(method,base,state,codes):
 arr=[base]+state+[codes];meta={'method':method,'shapes':[list(x.shape) for x in arr],'dtype':'float32','version':1};j=json.dumps(meta,sort_keys=True,separators=(',',':')).encode();return b'MA342\0'+struct.pack('<I',len(j))+j+b''.join(np.asarray(x,dtype=np.float32).tobytes() for x in arr)
def run(phase):
 rows=[]
 for seed in (DEV if phase=='development' else FRESH):
  a=world(seed);desc=np.stack([np.cos(a['phte']),np.sin(a['phte'])],-1).astype(np.float32);phasecode=a['phte'][:,None].astype(np.float32);zero=np.zeros((8,0),np.float32)
  results={}
  for name,fn,codes in [('mirror_phase',lambda:basis(a,True),phasecode),('generic_coefficients',lambda:basis(a,False),desc),('hyperlora_generator',lambda:hyper(a),desc),('independent_client_lora',lambda:independent(a),zero),('base_only',lambda:shared(a),zero)]:
   mse,state,sec,trainloss=fn();blob=pack(name,a['base'],state,codes);rows.append({'phase':phase,'world':seed,'method':name,'serialized_bytes':len(blob),'train_examples':CT*NTR,'optimizer_updates':UPDATES if name not in ('independent_client_lora','base_only') else 0,'active_compute_proxy':UPDATES*CT*NTR*D*O if name not in ('independent_client_lora','base_only') else 8*NTR*D*O,'wall_time_s':sec,'heldout_test_mse':mse,'train_mse':trainloss,'client_code_bytes':codes.nbytes,'payload_sha256':hashlib.sha256(blob).hexdigest()})
 path=ROOT/f'{phase.upper()}_RESULTS.csv'
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows),'seeds':DEV if phase=='development' else FRESH}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
