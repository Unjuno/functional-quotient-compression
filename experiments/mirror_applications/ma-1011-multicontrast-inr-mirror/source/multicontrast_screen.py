"""Analytic multi-contrast MRI INR screen; no patient data or third-party code."""
from __future__ import annotations
import hashlib,io,json,math,pathlib,time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT=pathlib.Path(__file__).resolve().parents[1];RUNS=ROOT/'runs';RUNS.mkdir(exist_ok=True)
torch.set_num_threads(1);SEED=1011001;H=W=24;N=8;STEPS=300
TIS=torch.tensor([.100,.350,.600,.850,1.100,1.350,1.600,1.850],dtype=torch.float32)

def truth():
 y,x=torch.meshgrid(torch.linspace(-1,1,H),torch.linspace(-1,1,W),indexing='ij');r=torch.sqrt((x/.86)**2+(y/.94)**2)
 brain=(r<1).float();wm=(r<.32).float();gm=((r>=.32)&(r<.78)).float();csf=((r>=.78)&(r<1)).float()
 t1=wm*.85+gm*1.35+csf*4.0;rho=wm*.78+gm*.90+csf*.98
 ims=[]
 for t in TIS:
  sig=rho*(1.-2.*torch.exp(-t/(t1+1e-4)))*brain
  # Weak non-concentric tissue variation makes the anatomy non-radially trivial.
  sig=sig + .035*torch.sin(5*x+2*y)*torch.exp(-((r-.52)/.13)**2)*(t/.9).clamp(0,2)
  ims.append(sig)
 return torch.stack(ims)

def masks():
 rng=np.random.default_rng(SEED+19);out=[];train=[];held=[]
 for n in range(N):
  lines=[11,12]; pool=[j for j in range(W) if j not in lines]
  lines+=list(rng.choice(pool,4,replace=False));m=np.zeros((H,W),bool);m[:,lines]=True
  hline=int(rng.choice(lines[2:]));hm=np.zeros_like(m);hm[:,hline]=True;tm=m&~hm
  out.append(m);train.append(tm);held.append(hm)
 return tuple(torch.tensor(np.stack(q)) for q in (out,train,held))

def kfft(im):return torch.fft.fft2(im,norm='ortho')
def coords():
 y,x=torch.meshgrid(torch.linspace(-1,1,H),torch.linspace(-1,1,W),indexing='ij');freq=torch.tensor([1.,2.,4.,8.])
 return torch.cat([torch.sin(math.pi*freq[:,None,None]*x),torch.cos(math.pi*freq[:,None,None]*x),torch.sin(math.pi*freq[:,None,None]*y),torch.cos(math.pi*freq[:,None,None]*y)],dim=0).permute(1,2,0).reshape(-1,16)

class NativeTimeINR(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(17,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
 def forward(self,xy,t):return self.net(torch.cat([xy,torch.full((len(xy),1),float(t))],dim=1)).reshape(H,W)
class SharedINR(nn.Module):
 def __init__(self):
  super().__init__();self.a=nn.Linear(16,64);self.b=nn.Linear(64,64);self.c=nn.Linear(64,4);self.head=nn.Linear(4,1)
 def features(self,xy):return self.c(F.relu(self.b(F.relu(self.a(xy)))))
 def forward(self,xy,view=None,mode='mirror'):
  z=self.features(xy)
  if view is not None:
   if mode=='film': z=z*(1+view[:4])+view[4:]
   elif mode=='mirror':
    a,b=view[0],view[1];ca,sa=torch.cos(a),torch.sin(a);cb,sb=torch.cos(b),torch.sin(b);u=z.clone()
    u[:,0]=ca*z[:,0]-sa*z[:,1];u[:,1]=sa*z[:,0]+ca*z[:,1];u[:,2]=cb*z[:,2]-sb*z[:,3];u[:,3]=sb*z[:,2]+cb*z[:,3];z=u
  return self.head(z).reshape(H,W)

def make_data():
 torch.manual_seed(SEED); x=truth();full=kfft(x);m,tr,ho=masks();return x,full,m,tr,ho,coords()
def fit_native(x,k,tr,xy):
 torch.manual_seed(SEED+1);m=NativeTimeINR();o=torch.optim.Adam(m.parameters(),lr=.002);t=time.perf_counter()
 for _ in range(STEPS):
  o.zero_grad();loss=0
  for j in range(N):
   pred=kfft(m(xy,TIS[j]));loss+=(((pred-k[j])*tr[j]) .abs()**2).sum()/tr[j].sum()
  (loss/N).backward();o.step()
 return m,time.perf_counter()-t

def fit_view(x,k,tr,xy,mode):
 torch.manual_seed(SEED+2);m=SharedINR();dim=8 if mode=='film' else 2;codes=nn.Parameter(torch.zeros(N,dim));o=torch.optim.Adam(list(m.parameters())+[codes],lr=.002);t=time.perf_counter()
 for _ in range(STEPS):
  o.zero_grad();loss=0
  for j in range(N):
   pred=kfft(m(xy,codes[j],mode));loss+=(((pred-k[j])*tr[j]).abs()**2).sum()/tr[j].sum()
  (loss/N).backward();o.step()
 return m,codes.detach(),time.perf_counter()-t

def fit_independent(x,k,tr,xy):
 pred=[];states=[];wall=0
 for j in range(N):
  torch.manual_seed(SEED+100+j);m=NativeTimeINR();o=torch.optim.Adam(m.parameters(),lr=.002);t=time.perf_counter()
  for _ in range(STEPS):
   o.zero_grad();p=kfft(m(xy,TIS[j]));loss=((p-k[j])*tr[j]).abs().square().sum()/tr[j].sum();loss.backward();o.step()
  wall+=time.perf_counter()-t
  with torch.no_grad():pred.append(m(xy,TIS[j]))
  states.append(m.state_dict())
 return torch.stack(pred),states,wall

def stats(p,x,k,ho):
 err=(p-x).square().mean().item();nm=err/(x.square().mean().item()+1e-12);ps=10*math.log10(4/(err+1e-12))
 # Global SSIM over each contrast, then average (range fixed by synthetic signal range).
 ss=[]
 for a,b in zip(p,x):
  ma,mb=a.mean(),b.mean();va=a.var(unbiased=False);vb=b.var(unbiased=False);cov=((a-ma)*(b-mb)).mean();c1=.01**2;c2=.03**2
  ss.append(float(((2*ma*mb+c1)*(2*cov+c2))/((ma.square()+mb.square()+c1)*(va+vb+c2))))
 res=[]
 for i in range(N):
  q=(kfft(p[i])-k[i])*ho[i];res.append(float((q.abs().square().sum()/(ho[i].sum()+1e-12)).sqrt()))
 return {'nmse':nm,'psnr_db':ps,'global_ssim':float(np.mean(ss)),'heldout_kspace_rmse':float(np.mean(res))}

def payload(model,method,codes=None,independent=None):
 obj={'schema':'MA1011-common-v1','architecture':'spatial-INR shared latent4; contrast-conditioned scalar output','forward_model':'orthonormal 2D FFT; 8 complementary Cartesian masks','method':'MA1011-inference-payload','metadata':{'TI_s':TIS,'shape':[H,W],'mask_fraction':6/24},'model':None,'contrast_codes':None,'independent_models':None}
 if model is not None: obj['model']=model.state_dict() if hasattr(model,'state_dict') else model
 if codes is not None:obj['contrast_codes']=codes
 if independent is not None:obj['independent_models']=independent
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()

def run():
 torch.manual_seed(SEED);x,k,m,tr,ho,xy=make_data();results={};payloads={};t0=time.perf_counter()
 nat,nw=fit_native(x,k,tr,xy)
 with torch.no_grad():pn=torch.stack([nat(xy,TIS[j]) for j in range(N)])
 results['native_time']=stats(pn,x,k,ho);results['native_time']['updates']=STEPS;results['native_time']['wall_s']=nw;payloads['native_time']=len(payload(nat,'native_time'))
 for mode in ('film','mirror'):
  mod,codes,wall=fit_view(x,k,tr,xy,mode)
  with torch.no_grad():pred=torch.stack([mod(xy,codes[j],mode) for j in range(N)])
  results[mode]=stats(pred,x,k,ho);results[mode].update({'updates':STEPS,'wall_s':wall,'code_dimensions_per_contrast':8 if mode=='film' else 2});payloads[mode]=len(payload(mod,mode,codes))
 ind,states,iw=fit_independent(x,k,tr,xy);results['independent']=stats(ind,x,k,ho);results['independent'].update({'updates':STEPS*N,'wall_s':iw});payloads['independent']=len(payload(None,'independent',independent=states))
 code=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest();d={'schema':'MA1011-dev-v1','code_sha256':code,'seed':SEED,'fresh_seeds':[1011007,1011013,1011019],'fresh_opened':False,'training':{'shared_methods_updates':STEPS,'independent_updates':STEPS*N,'samples':N*H*W,'mask_lines_per_contrast':6,'train_lines':5,'heldout_lines':1},'results':results,'payload_bytes':payloads,'wall_total_s':time.perf_counter()-t0}
 (RUNS/'development.json').write_text(json.dumps(d,indent=2)+'\n');torch.save({'native':nat.state_dict(),'shared':mod.state_dict()},RUNS/'development_models.pt');print(json.dumps(d,indent=2))
if __name__=='__main__':run()
