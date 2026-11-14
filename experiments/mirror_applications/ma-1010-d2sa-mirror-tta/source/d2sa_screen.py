"""Independent compact synthetic MRI TTA screen; no official D2SA code copied."""
from __future__ import annotations
import json, math, pathlib, time, io, hashlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=pathlib.Path(__file__).resolve().parents[1]; RUNS=ROOT/'runs'; RUNS.mkdir(exist_ok=True)
SEED=1010001; torch.set_num_threads(1); torch.manual_seed(SEED); np.random.seed(SEED)
DEV_SEEDS=list(range(1010201,1010209)); SUPPORT_SEEDS=list(range(1010101,1010105)); H=W=32; N=8; UPDATES=12

def phantom(seed, target=False):
    g=torch.Generator().manual_seed(seed); yy,xx=torch.meshgrid(torch.linspace(-1,1,H),torch.linspace(-1,1,W),indexing='ij'); out=torch.zeros(H,W)
    count=4 if not target else 5
    for j in range(count):
        cx=torch.rand((),generator=g).item()*(.8 if not target else 1.0)-.4
        cy=torch.rand((),generator=g).item()*(.8 if not target else 1.0)-.4
        sx=.08+.18*torch.rand((),generator=g).item(); sy=.08+.18*torch.rand((),generator=g).item()
        amp=.2+.8*torch.rand((),generator=g).item()
        if target: cx=min(.85,max(-.85,cx+.1)); sy*=.75
        out += amp*torch.exp(-((xx-cx)**2/sx**2+(yy-cy)**2/sy**2)/2)
    out/=out.max().clamp_min(1e-6)
    phase=.12*torch.sin(2*xx+seed*.001)*torch.cos(3*yy)
    return torch.stack([out*phase.cos(),out*phase.sin()],0)

def mask(seed, fraction=.25):
    # Cartesian lines: full readout, randomly choose 8 of 32 phase-encode lines; include center 2.
    g=np.random.default_rng(seed); lines=np.zeros(W,bool); lines[W//2-1:W//2+1]=True
    cand=np.array([i for i in range(W) if not lines[i]])
    lines[g.choice(cand,size=6,replace=False)]=True
    return torch.tensor(np.tile(lines[None,:],(H,1)),dtype=torch.bool)

def fft(x):
    z=torch.complex(x[0],x[1]); k=torch.fft.fft2(z,norm='ortho'); return torch.stack([k.real,k.imag])
def ifft(k):
    z=torch.complex(k[0],k[1]); x=torch.fft.ifft2(z,norm='ortho'); return torch.stack([x.real,x.imag])

def sample(x,seed):
    m=mask(seed); k=fft(x); keep=m.clone(); ids=torch.nonzero(m,as_tuple=False); g=torch.Generator().manual_seed(seed+123)
    hold=ids[torch.randperm(len(ids),generator=g)[:max(1,len(ids)//5)]]
    held=torch.zeros_like(m); held[hold[:,0],hold[:,1]]=True; train=keep & ~held
    return k,m,train,held,ifft(k*train)

class Prior(nn.Module):
    def __init__(self):
        super().__init__(); self.a=nn.Conv2d(2,8,3,padding=1); self.b=nn.Conv2d(8,4,3,padding=1); self.c=nn.Conv2d(4,2,3,padding=1)
    def feat(self,x):return F.silu(self.b(F.silu(self.a(x))))
    def forward(self,x, code=None, kind='base'):
        z=self.feat(x)
        if code is not None:
            if kind=='film':
                gamma,beta=code[:4],code[4:]; z=z*(1+gamma[None,:,None,None])+beta[None,:,None,None]
            elif kind=='mirror':
                a,b=code[0],code[1]; ca,sa=torch.cos(a),torch.sin(a); cb,sb=torch.cos(b),torch.sin(b)
                z=z.clone(); x0,x1=z[:,0].clone(),z[:,1].clone(); z[:,0]=ca*x0-sa*x1;z[:,1]=sa*x0+ca*x1
                x2,x3=z[:,2].clone(),z[:,3].clone();z[:,2]=cb*x2-sb*x3;z[:,3]=sb*x2+cb*x3
        return self.c(z)+x

def batch_data(seeds,target=False):
    xs=[]; ks=[]; ms=[]; tr=[]; ho=[]; zs=[]
    for i,s in enumerate(seeds):
        x=phantom(s,target); k,m,t,h,z=sample(x,s+8000);xs.append(x);ks.append(k);ms.append(m);tr.append(t);ho.append(h);zs.append(z)
    return [torch.stack(v) for v in (xs,ks,ms,tr,ho,zs)]

def metric(pred,truth):
    mse=((pred-truth)**2).mean().item(); nm=mse/(truth.square().mean().item()+1e-12)
    return nm,10*math.log10(1/(nm+1e-12))

def adapt_volume(model, data, mode, updates=UPDATES):
    xs,ks,ms,tr,ho,zf=data; started=time.perf_counter()
    dim=8 if mode=='film' else 2
    dist=nn.Parameter(torch.zeros(dim)); codes=nn.Parameter(torch.zeros(len(xs),dim)); opt=torch.optim.Adam([dist,codes],lr=.01)
    for _ in range(updates):
        opt.zero_grad(); loss=0
        for i in range(len(xs)):
            code=dist+codes[i]
            pred=fft(model(zf[i:i+1],code,mode)[0])
            loss=loss+(((pred-ks[i])*tr[i])**2).sum()/(tr[i].sum()*2)
        (loss/len(xs)).backward();opt.step()
    return dist.detach(),codes.detach(),time.perf_counter()-started

def adapt_slice(model, data, mode, dist, updates=UPDATES):
    xs,ks,ms,tr,ho,zf=data; started=time.perf_counter();dim=8 if mode=='film' else 2
    code=nn.Parameter(torch.zeros(dim));opt=torch.optim.Adam([code],lr=.01)
    for _ in range(updates):
        opt.zero_grad();pred=fft(model(zf[0:1],dist+code,mode)[0]);loss=(((pred-ks[0])*tr[0])**2).sum()/(tr[0].sum()*2);loss.backward();opt.step()
    return (dist+code).detach(),time.perf_counter()-started

def adapt_independent(model,data,updates=UPDATES):
    xs,ks,ms,tr,ho,zf=data; outs=[];start=time.perf_counter()
    for i in range(len(xs)):
        # Upper control updates all prior parameters on each single slice (same updates).
        m=Prior();m.load_state_dict(model.state_dict());opt=torch.optim.Adam(m.parameters(),lr=.0005)
        for _ in range(updates):
            opt.zero_grad(); pred=fft(m(zf[i:i+1])[0]);loss=(((pred-ks[i])*tr[i])**2).sum()/(tr[i].sum()*2);loss.backward();opt.step()
        outs.append(m.state_dict())
    return outs,time.perf_counter()-start

def serialize(model, method, dist=None,codes=None,indep=None):
    pkg={'schema':'MA1010-v1','arch':'32x32-2-8-4-2-cnn','operator':'single-coil-orthonormal-FFT','model':model.state_dict(),'method':method,'metadata':{'size':[32,32],'mask':'8 cartesian lines','updates_per_stage':12}}
    if dist is not None:pkg['distribution_code']=dist;pkg['slice_codes']=codes
    if indep is not None:pkg['independent_slice_models']=indep
    b=io.BytesIO();torch.save(pkg,b);return b.getvalue()

def run():
    t0=time.perf_counter(); torch.manual_seed(SEED)
    model=Prior();opt=torch.optim.Adam(model.parameters(),lr=.003)
    src=[]
    for s in range(1010001,1010033):
        x=phantom(s,False); _,_,_,_,z=sample(x,s+4000);src.append((z,x))
    for _ in range(80):
        ids=np.random.randint(len(src),size=8); pred=model(torch.stack([src[j][0] for j in ids])); y=torch.stack([src[j][1] for j in ids]); loss=F.mse_loss(pred,y);opt.zero_grad();loss.backward();opt.step()
    pretrain_s=time.perf_counter()-t0
    sup=batch_data(SUPPORT_SEEDS,True); dev=batch_data(DEV_SEEDS,True)
    results={}; payload={}
    with torch.no_grad():
        frozen=[model(dev[5][i:i+1])[0] for i in range(len(DEV_SEEDS))]
    results['frozen']={'nmse':float(np.mean([metric(frozen[i],dev[0][i])[0] for i in range(len(DEV_SEEDS))])),'psnr_db':float(np.mean([metric(frozen[i],dev[0][i])[1] for i in range(len(DEV_SEEDS))])),'updates':0,'wall_s':0,'heldout_measurement_residual':float(np.mean([((((fft(frozen[i])-dev[1][i])*dev[4][i])**2).sum()/(dev[4][i].sum()*2+1e-12)).sqrt().item() for i in range(len(DEV_SEEDS))]))}
    for mode in ('film','mirror'):
        # Shared distribution code learned from support; slice codes are adapted independently on dev samples.
        ds,sc,sw=adapt_volume(model,sup,mode)
        pred=[]; times=sw; allcodes=[]
        for i in range(len(DEV_SEEDS)):
            one=[v[i:i+1] for v in dev]
            code,w=adapt_slice(model,one,mode,ds)
            allcodes.append(code.clone());times+=w
            with torch.no_grad(): pred.append(model(dev[5][i:i+1],code,mode)[0])
        results[mode]={'nmse':float(np.mean([metric(pred[i],dev[0][i])[0] for i in range(len(DEV_SEEDS))])),'psnr_db':float(np.mean([metric(pred[i],dev[0][i])[1] for i in range(len(DEV_SEEDS))])),'updates':UPDATES*(1+len(DEV_SEEDS)),'wall_s':times,'code_dims':{'distribution':8 if mode=='film' else 2,'slice':8 if mode=='film' else 2},'stage_one_loss':'self-supervised acquired train partition','heldout_measurement_residual':float(np.mean([((((fft(pred[i])-dev[1][i])*dev[4][i])**2).sum()/(dev[4][i].sum()*2+1e-12)).sqrt().item() for i in range(len(DEV_SEEDS))]))}
        payload[mode]=len(serialize(model,mode,ds,torch.stack(allcodes)))
    indep,iw=adapt_independent(model,dev)
    with torch.no_grad():
        # evaluate upper reconstructed prior per case using adapted state
        ip=[]
        for i,st in enumerate(indep):
            mi=Prior();mi.load_state_dict(st);ip.append(mi(dev[5][i:i+1])[0])
    results['independent']={'nmse':float(np.mean([metric(ip[i],dev[0][i])[0] for i in range(len(DEV_SEEDS))])),'psnr_db':float(np.mean([metric(ip[i],dev[0][i])[1] for i in range(len(DEV_SEEDS))])),'updates':UPDATES*len(DEV_SEEDS),'wall_s':iw,'heldout_measurement_residual':float(np.mean([((((fft(ip[i])-dev[1][i])*dev[4][i])**2).sum()/(dev[4][i].sum()*2+1e-12)).sqrt().item() for i in range(len(DEV_SEEDS))]))}
    payload['frozen']=len(serialize(model,'frozen'));payload['independent']=len(serialize(model,'independent',indep=indep))
    # heldout residual measured at model prediction after separate view adaptation; expected nonzero; values are scored only.
    result={'schema':'MA1010-dev-v1','code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'seed_protocol':{'source_train':list(range(1010001,1010033)),'support':SUPPORT_SEEDS,'development':DEV_SEEDS,'fresh':[1010301,1010307,1010313],'fresh_opened':False},'pretrain':{'updates':80,'examples':32,'wall_s':pretrain_s,'trainable_parameters':sum(p.numel() for p in model.parameters())},'development_metrics':results,'payload_bytes':payload,'wall_total_s':time.perf_counter()-t0}
    (RUNS/'development.json').write_text(json.dumps(result,indent=2)+'\n')
    (RUNS/'development_models.pt').write_bytes(serialize(model,'source_checkpoint'))
    print(json.dumps(result,indent=2))
if __name__=='__main__':run()
