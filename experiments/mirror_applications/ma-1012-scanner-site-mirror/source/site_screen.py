"""Independent synthetic 2-coil site-adaptation screen; no VarNet code copied."""
from __future__ import annotations
import hashlib,io,json,math,pathlib,time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT=pathlib.Path(__file__).resolve().parents[1];RUNS=ROOT/'runs';RUNS.mkdir(exist_ok=True)
torch.set_num_threads(1);SRC=1012001;SUPPORT=[[1012101,1012102,1012103,1012104],[1012111,1012112,1012113,1012114],[1012121,1012122,1012123,1012124]];DEV=[[1012201+i for i in range(8)],[1012211+i for i in range(8)],[1012221+i for i in range(8)]];H=W=32;UPDATES=48

def phantom(seed,target=False):
 g=torch.Generator().manual_seed(seed);y,x=torch.meshgrid(torch.linspace(-1,1,H),torch.linspace(-1,1,W),indexing='ij');out=torch.zeros(H,W);count=4 if not target else 6
 for j in range(count):
  cx=(torch.rand((),generator=g).item()-.5)*(.8 if not target else 1.1);cy=(torch.rand((),generator=g).item()-.5)*(.8 if not target else 1.1)
  sx=.10+.18*torch.rand((),generator=g).item();sy=.08+.20*torch.rand((),generator=g).item();amp=.2+.8*torch.rand((),generator=g).item()
  out+=amp*torch.exp(-(((x-cx)/sx)**2+((y-cy)/sy)**2)/2)
 out/=out.max().clamp_min(1e-8);phase=.12*torch.sin(2*x+seed*.0001)*torch.cos(3*y)
 return torch.stack([out*torch.cos(phase),out*torch.sin(phase)])

def sens(site):
 y,x=torch.meshgrid(torch.linspace(-1,1,H),torch.linspace(-1,1,W),indexing='ij');th=math.pi/4+site*.12+.09*torch.sin((1+site*.3)*math.pi*x)*torch.cos(math.pi*y)
 p1=site*.32+.18*x+.05*y;p2=-site*.25+.14*y-.04*x
 s1=torch.polar(torch.cos(th),p1);s2=torch.polar(torch.sin(th),p2);return torch.stack([s1,s2])

def mask(seed,site):
 rng=np.random.default_rng(seed+9000+site*123);lines=[15,16,14,17];pool=[i for i in range(W) if i not in lines];lines+=list(rng.choice(pool,4,replace=False));m=np.zeros((H,W),bool);m[:,lines]=True
 holdline=int(rng.choice(lines[4:]));h=np.zeros_like(m);h[:,holdline]=True;tr=m&~h
 return tuple(torch.tensor(v) for v in (m,tr,h))

def img2(x):return torch.complex(x[0],x[1])
def fft2(x):return torch.fft.fft2(x,norm='ortho')
def ifft2(k):return torch.fft.ifft2(k,norm='ortho')
def A(x,s,m):return fft2(s*img2(x)[None])*m[None]
def AH(k,s):return (s.conj()*ifft2(k)).sum(0)
def zf(k,s):return torch.stack([AH(k,s).real,AH(k,s).imag])

def make_case(seed,site):
 x=phantom(seed,target=site>0);s=sens(site);m,tr,ho=mask(seed,site);k=A(x,s,m);return {'x':x,'s':s,'m':m,'tr':tr,'ho':ho,'k':k,'zf':zf(k*tr[None],s)}

class Prior(nn.Module):
 def __init__(self):
  super().__init__();self.a=nn.Conv2d(2,8,3,padding=1);self.b=nn.Conv2d(8,4,3,padding=1);self.c=nn.Conv2d(4,2,3,padding=1)
 def feat(self,x):return F.silu(self.b(F.silu(self.a(x))))
 def forward(self,x,code=None,mode='base'):
  z=self.feat(x)
  if code is not None:
   if mode=='film':z=z*(1+code[:4][None,:,None,None])+code[4:][None,:,None,None]
   elif mode=='mirror':
    a,b=code;ca,sa=torch.cos(a),torch.sin(a);cb,sb=torch.cos(b),torch.sin(b);u=z.clone()
    u[:,0]=ca*z[:,0]-sa*z[:,1];u[:,1]=sa*z[:,0]+ca*z[:,1];u[:,2]=cb*z[:,2]-sb*z[:,3];u[:,3]=sb*z[:,2]+cb*z[:,3];z=u
  return self.c(z)+x

def adapt_site(model,cases,mode):
 dim=8 if mode=='film' else 2;code=nn.Parameter(torch.zeros(dim));opt=torch.optim.Adam([code],lr=.01);t=time.perf_counter()
 for _ in range(UPDATES):
  opt.zero_grad();loss=0
  for c in cases:
   pred=model(c['zf'][None],code,mode)[0];kpred=A(pred,c['s'],c['tr']);loss+=((kpred-c['k']*c['tr'][None]).abs().square().sum()/(c['tr'].sum()*2))
  (loss/len(cases)).backward();opt.step()
 return code.detach(),time.perf_counter()-t

def finetune_site(model,cases):
 m=Prior();m.load_state_dict(model.state_dict());opt=torch.optim.Adam(m.parameters(),lr=.0005);t=time.perf_counter()
 for it in range(UPDATES):
  c=cases[it%len(cases)];opt.zero_grad();pred=m(c['zf'][None])[0];kpred=A(pred,c['s'],c['tr']);loss=(kpred-c['k']*c['tr'][None]).abs().square().sum()/(c['tr'].sum()*2);loss.backward();opt.step()
 return m.state_dict(),time.perf_counter()-t

def metrics(pred,c):
 x=c['x'];err=(pred-x).square().mean().item();nm=err/(x.square().mean().item()+1e-12);ps=10*math.log10(4/(err+1e-12));s1=c['s'];hold=(A(pred,s1,c['ho'])-c['k']*c['ho'][None]).abs().square().sum()/(c['ho'].sum()*2+1e-12)
 # Global SSIM proxy on magnitude images, scored only after all fitting is complete.
 a=pred.square().sum(0).sqrt();b=x.square().sum(0).sqrt();ma=a.mean();mb=b.mean();va=a.var(unbiased=False);vb=b.var(unbiased=False);cov=((a-ma)*(b-mb)).mean();c1=.01**2;c2=.03**2
 ssim=float(((2*ma*mb+c1)*(2*cov+c2))/((ma.square()+mb.square()+c1)*(va+vb+c2)))
 return nm,ps,float(hold.sqrt()),ssim

def pack(base,codes=None,private=None,name='common'):
 pkg={'schema':'MA1012-common-v1','architecture':'32x32-2coil-sense-prior-2-8-4-2','method':'MA1012-deployment','metadata':{'sites':3,'shape':[H,W],'mask_protocol':'8 cartesian lines; center 4','operator':'known site-specific SENSE maps + orthonormal FFT'},'base':None if private is not None else base.state_dict(),'site_codes':None,'private_site_models':None}
 if codes is not None:pkg['site_codes']=codes
 if private is not None:pkg['private_site_models']=private
 b=io.BytesIO();torch.save(pkg,b);return b.getvalue()

def run():
 torch.manual_seed(SRC);np.random.seed(SRC);t0=time.perf_counter();model=Prior();opt=torch.optim.Adam(model.parameters(),lr=.003)
 src=[make_case(s,0) for s in range(1012001,1012017)]
 for it in range(64):
  ids=np.random.randint(len(src),size=4);pred=model(torch.stack([src[i]['zf'] for i in ids]));x=torch.stack([src[i]['x'] for i in ids]);loss=(pred-x).square().mean();opt.zero_grad();loss.backward();opt.step()
 pre_s=time.perf_counter()-t0
 sup=[[make_case(seed,site+1) for seed in SUPPORT[site]] for site in range(3)];dev=[[make_case(seed,site+1) for seed in DEV[site]] for site in range(3)]
 results={};predictions={};codepacks={};walls={}
 with torch.no_grad():
  predictions['frozen']=[[model(c['zf'][None])[0] for c in ss] for ss in dev]
 for mode in ('film','mirror'):
  preds=[];codes=[];wall=0
  for si in range(3):
   code,w=adapt_site(model,sup[si],mode);codes.append(code);wall+=w
   with torch.no_grad():preds.append([model(c['zf'][None],code,mode)[0] for c in dev[si]])
  predictions[mode]=preds;walls[mode]=wall;codepacks[mode]=torch.stack(codes)
 states=[];preds=[];wall=0
 for si in range(3):
  state,w=finetune_site(model,sup[si]);states.append(state);wall+=w;m=Prior();m.load_state_dict(state)
  with torch.no_grad():preds.append([m(c['zf'][None])[0] for c in dev[si]])
 predictions['independent']=preds;walls['independent']=wall
 sizes={'frozen':len(pack(model,name='frozen')),'film':len(pack(model,codepacks['film'])),'mirror':len(pack(model,codepacks['mirror'])),'independent':len(pack(model,private=states))}
 for method,groups in predictions.items():
  vals=[metrics(pred,c) for predgroup,cgroup in zip(groups,dev) for pred,c in zip(predgroup,cgroup)]
  results[method]={'nmse':float(np.mean([v[0] for v in vals])),'psnr_db':float(np.mean([v[1] for v in vals])),'heldout_kspace_rmse':float(np.mean([v[2] for v in vals])),'global_ssim_magnitude_proxy':float(np.mean([v[3] for v in vals])),'updates':0 if method=='frozen' else UPDATES*3,'wall_s':walls.get(method,0)}
 d={'schema':'MA1012-dev-v1','code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'source_seeds':list(range(1012001,1012017)),'support_seeds':SUPPORT,'development_seeds':DEV,'fresh_seeds':[1012301,1012307,1012313],'fresh_opened':False,'pretrain':{'updates':64,'examples':16,'wall_s':pre_s,'trainable_parameters':sum(p.numel() for p in model.parameters())},'results':results,'payload_bytes':sizes,'wall_total_s':time.perf_counter()-t0}
 (RUNS/'development.json').write_text(json.dumps(d,indent=2)+'\n');torch.save(model.state_dict(),RUNS/'development_models.pt');print(json.dumps(d,indent=2))
if __name__=='__main__':run()
