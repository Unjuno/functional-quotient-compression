"""Synthetic MRI projection-placement audit; never opens fresh seeds."""
from __future__ import annotations
import hashlib, io, json, math, pathlib, time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=pathlib.Path(__file__).resolve().parents[1]; RUNS=ROOT/'runs'; RUNS.mkdir(exist_ok=True)
torch.set_num_threads(1)
H=W=32; SOURCE=1014001; TARGETS=[[1014101+i for i in range(8)],[1014111+i for i in range(8)],[1014121+i for i in range(8)]]; UPDATES=64

def phantom(seed, target=False):
    g=torch.Generator().manual_seed(seed); y,x=torch.meshgrid(torch.linspace(-1,1,H),torch.linspace(-1,1,W),indexing='ij'); out=torch.zeros(H,W)
    for _ in range(6 if target else 4):
        cx=(torch.rand((),generator=g).item()-.5)*(.8 if not target else 1.1); cy=(torch.rand((),generator=g).item()-.5)*(.8 if not target else 1.1)
        sx=.10+.18*torch.rand((),generator=g).item(); sy=.08+.20*torch.rand((),generator=g).item(); amp=.2+.8*torch.rand((),generator=g).item()
        out+=amp*torch.exp(-(((x-cx)/sx)**2+((y-cy)/sy)**2)/2)
    out/=out.max().clamp_min(1e-8); phase=.12*torch.sin(2*x+seed*.0001)*torch.cos(3*y); return torch.stack([out*torch.cos(phase),out*torch.sin(phase)])

def sens(site):
    # Spatially constant unit-norm coil sensitivities make the SENSE normal
    # operator identity, so one hard k-space replacement is an exact projection.
    th=math.pi/4+site*.1
    return torch.stack([torch.full((H,W),complex(math.cos(th),0)),torch.full((H,W),complex(math.sin(th),0))])
def fft(x): return torch.fft.fft2(x,norm='ortho')
def ifft(x): return torch.fft.ifft2(x,norm='ortho')
def cplx(x): return torch.complex(x[0],x[1])
def A(x,s): return fft(s*cplx(x)[None])
def AH(k,s): return (s.conj()*ifft(k)).sum(0)
def zf(k,s):
    q=AH(k,s); return torch.stack([q.real,q.imag])
def mask(seed):
    rng=np.random.default_rng(seed+400); lines=[15,16,14,17]; lines += list(rng.choice([i for i in range(W) if i not in lines],4,replace=False)); m=torch.zeros(H,W,dtype=torch.bool); m[:,lines]=True; return m
def case(seed,site):
    x=phantom(seed,target=True); s=sens(site); m=mask(seed); full=A(x,s); k=full*m[None]; return {'x':x,'s':s,'m':m,'k':k,'full_k':full,'zf':zf(k,s)}

class Prior(nn.Module):
    def __init__(self):
        super().__init__(); self.a=nn.Conv2d(2,8,3,padding=1); self.b=nn.Conv2d(8,4,3,padding=1); self.c=nn.Conv2d(4,2,3,padding=1)
    def forward(self,x,mirror=False,angles=None):
        z=F.silu(self.b(F.silu(self.a(x))))
        if mirror:
            a,b=angles; u=z.clone(); u[:,0]=torch.cos(a)*z[:,0]-torch.sin(a)*z[:,1]; u[:,1]=torch.sin(a)*z[:,0]+torch.cos(a)*z[:,1]; u[:,2]=torch.cos(b)*z[:,2]-torch.sin(b)*z[:,3]; u[:,3]=torch.sin(b)*z[:,2]+torch.cos(b)*z[:,3]; z=u
        return self.c(z)+x

def project(x,c):
    pred=A(x,c['s']); y=pred.clone(); y[:,c['m']]=c['k'][:,c['m']]; return zf(y,c['s'])
def metrics(pred,c):
    predc=A(pred,c['s']); acq=(predc[:,c['m']]-c['k'][:,c['m']]).abs().square().mean().sqrt().item(); hold=~c['m']; held=(predc[:,hold]-c['full_k'][:,hold]).abs().square().mean().sqrt().item()
    nm=(pred-c['x']).square().mean().item()/(c['x'].square().mean().item()+1e-12); return acq,held,nm
def pack(model,placement,angles):
    payload={'schema':'MA1014-inference-v1','architecture':'32x32-2coil-sense-prior-2-8-4-2','prior':model.state_dict(),'mirror_angles':angles,'placement':placement,'operator':'known two-coil sensitivity + orthonormal FFT','mask_protocol':'8 Cartesian lines','dtype':'torch tensors'}
    f=io.BytesIO(); torch.save(payload,f); return f.getvalue()
def run():
    torch.manual_seed(SOURCE); np.random.seed(SOURCE); t0=time.perf_counter(); model=Prior(); opt=torch.optim.Adam(model.parameters(),lr=.003); src=[]
    for seed in range(SOURCE,SOURCE+16):
        x=phantom(seed); s=sens(0); m=mask(seed); k=A(x,s)*m[None]; src.append({'x':x,'zf':zf(k,s)})
    for _ in range(UPDATES):
        ids=np.random.randint(len(src),size=4); xx=torch.stack([src[i]['zf'] for i in ids]); yy=torch.stack([src[i]['x'] for i in ids]); opt.zero_grad(); loss=(model(xx)-yy).square().mean(); loss.backward(); opt.step()
    pretrain_s=time.perf_counter()-t0; model.requires_grad_(False); angle=torch.tensor([.37,-.29]); results=[]; counts=0; fw_time={}
    for site,seeds in enumerate(TARGETS,1):
        group=[case(seed,site) for seed in seeds]
        for name in ('plain_dc','prior_then_dc','mirror_before_prior_then_dc','mirror_after_dc','mirror_after_then_reproject'):
            vals=[]; st=time.perf_counter()
            for c in group:
                if name=='plain_dc': pred=c['zf']
                elif name=='prior_then_dc': pred=project(model(c['zf'][None])[0],c)
                elif name=='mirror_before_prior_then_dc': pred=project(model(c['zf'][None],True,angle)[0],c)
                elif name=='mirror_after_dc': pred=model(project(model(c['zf'][None])[0],c)[None],True,angle)[0]
                else: pred=project(model(project(model(c['zf'][None])[0],c)[None],True,angle)[0],c)
                vals.append(metrics(pred,c))
            fw_time[name]=fw_time.get(name,0)+time.perf_counter()-st
            mflag='mirror' in name; payload=len(pack(model,name,angle.tolist() if mflag else None));
            results.append({'site':site,'method':name,'acquired_residual':float(np.mean([v[0] for v in vals])),'unacquired_kspace_rmse':float(np.mean([v[1] for v in vals])),'nmse':float(np.mean([v[2] for v in vals])),'payload_bytes':payload,'updates':0,'eval_examples':len(group),'wall_s':fw_time[name]/site})
    d={'schema':'MA1014-dev-v1','source_seeds':list(range(SOURCE,SOURCE+16)),'target_seeds':TARGETS,'fresh_seeds':[1014301,1014307,1014313],'fresh_opened':False,'pretrain':{'updates':UPDATES,'unique_examples':16,'presentations':UPDATES*4,'wall_s':pretrain_s},'mirror_angles_radians':angle.tolist(),'results':results,'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'total_wall_s':time.perf_counter()-t0}
    (RUNS/'development.json').write_text(json.dumps(d,indent=2)+'\n'); torch.save(model.state_dict(),RUNS/'development_model.pt'); print(json.dumps(d,indent=2))
if __name__=='__main__': run()
