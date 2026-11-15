"""Synthetic SSDU-style scan-code test; held-out k-space and truth are score-only."""
from __future__ import annotations
import hashlib,io,json,math,pathlib,time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT=pathlib.Path(__file__).resolve().parents[1];RUNS=ROOT/'runs';RUNS.mkdir(exist_ok=True)
torch.set_num_threads(1);SRC=1013001;SCANS=[[1013101+i for i in range(8)],[1013111+i for i in range(8)],[1013121+i for i in range(8)]];H=W=32;UPDATES=48

def phantom(seed,target=False):
 g=torch.Generator().manual_seed(seed);y,x=torch.meshgrid(torch.linspace(-1,1,H),torch.linspace(-1,1,W),indexing='ij');out=torch.zeros(H,W);count=4 if not target else 6
 for _ in range(count):
  cx=(torch.rand((),generator=g).item()-.5)*(.8 if not target else 1.1);cy=(torch.rand((),generator=g).item()-.5)*(.8 if not target else 1.1);sx=.10+.18*torch.rand((),generator=g).item();sy=.08+.20*torch.rand((),generator=g).item();amp=.2+.8*torch.rand((),generator=g).item();out+=amp*torch.exp(-(((x-cx)/sx)**2+((y-cy)/sy)**2)/2)
 out/=out.max().clamp_min(1e-8);phase=.12*torch.sin(2*x+seed*.0001)*torch.cos(3*y);return torch.stack([out*torch.cos(phase),out*torch.sin(phase)])

def sens(site):
 y,x=torch.meshgrid(torch.linspace(-1,1,H),torch.linspace(-1,1,W),indexing='ij');th=math.pi/4+site*.12+.09*torch.sin((1+site*.3)*math.pi*x)*torch.cos(math.pi*y);p1=site*.32+.18*x+.05*y;p2=-site*.25+.14*y-.04*x;s1=torch.polar(torch.cos(th),p1);s2=torch.polar(torch.sin(th),p2);return torch.stack([s1,s2])

def masks(seed,site):
 rng=np.random.default_rng(seed+9000+site*123);lines=[15,16,14,17];lines+=list(rng.choice([i for i in range(W) if i not in lines],4,replace=False));m=np.zeros((H,W),bool);m[:,lines]=True;hold=int(rng.choice(lines[4:]));h=np.zeros_like(m);h[:,hold]=True;return tuple(torch.tensor(v) for v in (m,m&~h,h))
def img2(x):return torch.complex(x[0],x[1])
def fft2(x):return torch.fft.fft2(x,norm='ortho')
def ifft2(k):return torch.fft.ifft2(k,norm='ortho')
def A(x,s,m):return fft2(s*img2(x)[None])*m[None]
def AH(k,s):return (s.conj()*ifft2(k)).sum(0)
def zf(k,s):return torch.stack([AH(k,s).real,AH(k,s).imag])
def make_case(seed,site):
 x=phantom(seed,target=site>0);s=sens(site);m,tr,ho=masks(seed,site);k=A(x,s,m);return {'x':x,'s':s,'m':m,'tr':tr,'ho':ho,'k':k,'zf':zf(k*tr[None],s)}

class Prior(nn.Module):
 def __init__(self):
  super().__init__();self.a=nn.Conv2d(2,8,3,padding=1);self.b=nn.Conv2d(8,4,3,padding=1);self.c=nn.Conv2d(4,2,3,padding=1)
 def feat(self,x):return F.silu(self.b(F.silu(self.a(x))))
 def forward(self,x,code=None,mode='base'):
  z=self.feat(x)
  if code is not None:
   if mode=='film':z=z*(1+code[:4][None,:,None,None])+code[4:][None,:,None,None]
   elif mode=='mirror':
    a,b=code;ca,sa=torch.cos(a),torch.sin(a);cb,sb=torch.cos(b),torch.sin(b);u=z.clone();u[:,0]=ca*z[:,0]-sa*z[:,1];u[:,1]=sa*z[:,0]+ca*z[:,1];u[:,2]=cb*z[:,2]-sb*z[:,3];u[:,3]=sb*z[:,2]+cb*z[:,3];z=u
  out=self.c(z)+x
  if code is not None and mode=='affine':out=out*(1+code[0])+code[1]
  return out

def adapt_scan(model,scan,mode):
 dim={'affine':2,'film':8,'mirror':2}[mode];code=nn.Parameter(torch.zeros(dim));opt=torch.optim.Adam([code],lr=.01);t=time.perf_counter()
 for _ in range(UPDATES):
  opt.zero_grad();loss=0.
  for c in scan:
   pred=model(c['zf'][None],code,mode)[0];kp=A(pred,c['s'],c['tr']);loss+=((kp-c['k']*c['tr'][None]).abs().square().sum()/(c['tr'].sum()*2))
  (loss/len(scan)).backward();opt.step()
 return code.detach(),time.perf_counter()-t

def private_tune(model,scan):
 m=Prior();m.load_state_dict(model.state_dict());opt=torch.optim.Adam(m.parameters(),lr=.0005);t=time.perf_counter()
 for _ in range(UPDATES):
  opt.zero_grad();loss=0.
  for c in scan:
   pred=m(c['zf'][None])[0];kp=A(pred,c['s'],c['tr']);loss+=((kp-c['k']*c['tr'][None]).abs().square().sum()/(c['tr'].sum()*2))
  (loss/len(scan)).backward();opt.step()
 return m.state_dict(),time.perf_counter()-t

def metrics(pred,c):
 x=c['x'];err=(pred-x).square().mean().item();nm=err/(x.square().mean().item()+1e-12);ps=10*math.log10(4/(err+1e-12));q=(A(pred,c['s'],c['ho'])-c['k']*c['ho'][None]).abs().square().sum()/(c['ho'].sum()*2+1e-12);a=pred.square().sum(0).sqrt();b=x.square().sum(0).sqrt();ma=a.mean();mb=b.mean();va=a.var(unbiased=False);vb=b.var(unbiased=False);cov=((a-ma)*(b-mb)).mean();c1=.01**2;c2=.03**2;ss=float(((2*ma*mb+c1)*(2*cov+c2))/((ma.square()+mb.square()+c1)*(va+vb+c2)))
 return nm,ps,float(q.sqrt()),ss

def pack(base,codes=None,private=None,nscans=3):
 q={'schema':'MA1013-common-v1','architecture':'32x32-2coil-sense-prior-2-8-4-2','method':'MA1013-inference-payload','metadata':{'scan_count':nscans,'shape':[H,W],'mask_protocol':'8 Cartesian lines, four center','operator':'known site SENSE + orthonormal FFT'},'shared_prior':None if private is not None else base.state_dict(),'scan_codes':None,'private_models':None}
 if codes is not None:q['scan_codes']=codes
 if private is not None:q['private_models']=private
 f=io.BytesIO();torch.save(q,f);return f.getvalue()

def run():
 torch.manual_seed(SRC);np.random.seed(SRC);t0=time.perf_counter();base=Prior();opt=torch.optim.Adam(base.parameters(),lr=.003);src=[make_case(s,0) for s in range(1013001,1013017)]
 for _ in range(64):
  ids=np.random.randint(len(src),size=4);pred=base(torch.stack([src[i]['zf'] for i in ids]));x=torch.stack([src[i]['x'] for i in ids]);loss=(pred-x).square().mean();opt.zero_grad();loss.backward();opt.step()
 pre_s=time.perf_counter()-t0;base.requires_grad_(False);scans=[[make_case(seed,site+1) for seed in seeds] for site,seeds in enumerate(SCANS)];prediction={'frozen':[[base(c['zf'][None])[0].detach() for c in scan] for scan in scans]};codes={};priv={};walls={}
 for mode in ('affine','film','mirror'):
  cp=[];preds=[];wall=0
  for scan in scans:
   code,w=adapt_scan(base,scan,mode);cp.append(code);wall+=w;preds.append([base(c['zf'][None],code,mode)[0].detach() for c in scan])
  codes[mode]=torch.stack(cp);prediction[mode]=preds;walls[mode]=wall
 pp=[];states=[];wall=0
 for scan in scans:
  state,w=private_tune(base,scan);states.append(state);wall+=w;m=Prior();m.load_state_dict(state);pp.append([m(c['zf'][None])[0].detach() for c in scan])
 prediction['private']=pp;priv=states;walls['private']=wall
 results={};
 for method,groups in prediction.items():
  vals=[metrics(p,c) for pg,cg in zip(groups,scans) for p,c in zip(pg,cg)];results[method]={'nmse':float(np.mean([v[0] for v in vals])),'psnr_db':float(np.mean([v[1] for v in vals])),'heldout_kspace_rmse':float(np.mean([v[2] for v in vals])),'global_ssim_magnitude_proxy':float(np.mean([v[3] for v in vals])),'updates':0 if method=='frozen' else UPDATES*3,'wall_s':walls.get(method,0)}
 sizes={};single_sizes={}
 for method in ('frozen','affine','film','mirror','private'):
  cpack=codes.get(method);states=priv if method=='private' else None;single_codes=cpack[:1] if cpack is not None else None;single_states=states[:1] if states is not None else None
  sizes[method]=len(pack(base,cpack,states,3));single_sizes[method]=len(pack(base,single_codes,single_states,1))
 d={'schema':'MA1013-dev-v1','code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'source_seeds':list(range(1013001,1013017)),'scan_seeds':SCANS,'fresh_seeds':[1013301,1013307,1013313],'fresh_opened':False,'pretrain':{'updates':64,'unique_examples':16,'presentations':256,'wall_s':pre_s,'parameters':sum(p.numel() for p in base.parameters())},'adaptation':{'updates_per_scan':UPDATES,'scans':3,'slices_per_scan':8,'support_presentations_per_method':3*UPDATES*8,'loss_reads_only':['zf','s','train_mask','kspace'],'never_loss_reads':['fully_sampled_truth','heldout_mask']},'results':results,'payload_bytes_three_scans':sizes,'payload_bytes_one_scan':single_sizes,'wall_total_s':time.perf_counter()-t0}
 (RUNS/'development.json').write_text(json.dumps(d,indent=2)+'\n');torch.save(base.state_dict(),RUNS/'development_models.pt');print(json.dumps(d,indent=2))
if __name__=='__main__':run()
