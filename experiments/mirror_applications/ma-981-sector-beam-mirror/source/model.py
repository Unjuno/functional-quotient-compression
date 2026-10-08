"""Phase-only ULA codebooks and sector-view models for MA-981."""
import hashlib
import math
from pathlib import Path
import torch
from torch import nn

METHODS=('dft','global_learned','scalar_view','rank2_view','mirror_view','independent_sector')
NANT,NCODE,NSECTOR,NRAY=8,8,4,3
CENTERS=(-45.,-15.,15.,45.)
SNR=10.0


def steering_phases():
    n=torch.arange(NANT).float()
    vals=torch.linspace(-.875,.875,NCODE)
    angles=torch.arcsin(vals)
    return math.pi*torch.sin(angles)[:,None]*n[None,:]


def make_channels(seed,split,n):
    offset={'train':0,'development':1_000_000,'fresh':2_000_000}[split]
    g=torch.Generator().manual_seed(seed+offset)
    out=[];ant=torch.arange(NANT).float()
    for center in CENTERS:
        angle=center+3.*torch.randn(n,NRAY,generator=g)
        real=torch.randn(n,NRAY,generator=g);imag=torch.randn(n,NRAY,generator=g)
        gain=torch.complex(real,imag)/math.sqrt(2*NRAY)
        phase=math.pi*torch.sin(torch.deg2rad(angle))[:,:,None]*ant[None,None,:]
        a=torch.polar(torch.ones_like(phase)/math.sqrt(NANT),phase)
        h=(gain[:,:,None]*a).sum(1)
        out.append(h)
    return torch.stack(out)


class BeamCodebook(nn.Module):
    def __init__(self,method,seed):
        super().__init__();self.method=method
        base=steering_phases()
        if method in {'dft','global_learned','scalar_view','rank2_view','mirror_view'}:
            self.base=nn.Parameter(base.clone(),requires_grad=method!='dft')
        if method=='scalar_view':self.offset=nn.Parameter(torch.zeros(NSECTOR,1))
        elif method=='mirror_view':self.offset=nn.Parameter(torch.zeros(NSECTOR,NANT))
        elif method=='rank2_view':
            self.offset=nn.Parameter(torch.zeros(NSECTOR,2))
            gen=torch.Generator().manual_seed(seed+17)
            self.basis=nn.Parameter(torch.randn(2,NANT,generator=gen)*.03)
        elif method=='independent_sector':
            self.phase=nn.Parameter(base[None,:,:].repeat(NSECTOR,1,1).clone())
        elif method not in {'dft','global_learned'}:raise ValueError(method)

    def phases(self,sector):
        if self.method in {'dft','global_learned'}:return self.base
        if self.method=='independent_sector':return self.phase[sector]
        if self.method in {'scalar_view','mirror_view'}:return self.base+self.offset[sector]
        return self.base+self.offset[sector]@self.basis

    def beams(self,sector):
        p=self.phases(sector)
        return torch.complex(torch.cos(p),torch.sin(p))/math.sqrt(NANT)

    def active_proxy(self):
        return {'dft':0,'global_learned':0,'scalar_view':8,'rank2_view':24,'mirror_view':8,'independent_sector':0}[self.method]


def rates(channels,beams):
    gain=channels@beams.conj().T
    power=gain.abs().square()
    return torch.log2(1+SNR*power)


def save_payload(path,model,seed):
    if model.method=='dft':state={'base':model.base.detach().cpu().contiguous()}
    else:state={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    p={'schema':'MA-981/inference-codebook-v1','seed':seed,'method':model.method,
       'array':{'elements':NANT,'spacing_wavelengths':.5,'rf_chains':1,'phase_only':True},
       'sectors':list(CENTERS),'codewords':NCODE,'feedback_bits':3,'snr_db':10,
       'state_dict':state}
    torch.save(p,path);raw=path.read_bytes()
    return {'path':str(path.name),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
