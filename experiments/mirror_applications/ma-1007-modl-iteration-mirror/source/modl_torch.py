"""Independent compact PyTorch MoDL implementation for MA-1007 synthetic screens."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


def fft2c(x):
    x=torch.fft.ifftshift(x,dim=(-2,-1))
    return torch.fft.fftshift(torch.fft.fft2(x,norm='ortho'),dim=(-2,-1))
def ifft2c(x):
    x=torch.fft.ifftshift(x,dim=(-2,-1))
    return torch.fft.fftshift(torch.fft.ifft2(x,norm='ortho'),dim=(-2,-1))

def forward_op(x,sens,mask):return fft2c(sens*x)*mask
def adjoint_op(y,sens,mask):return (sens.conj()*ifft2c(y*mask)).sum(dim=0)
def normal_op(x,sens,mask):return adjoint_op(forward_op(x,sens,mask),sens,mask)

def make_case(seed:int,scanner:str='A',size:int=32,coils:int=4,accel:int=4,center:int=8):
    g=torch.Generator().manual_seed(seed)
    yy,xx=torch.meshgrid(torch.linspace(-1,1,size),torch.linspace(-1,1,size),indexing='ij')
    image=torch.zeros(size,size)
    for _ in range(6):
        vals=torch.rand(5,generator=g)
        cx=float(vals[0]-.5)*.9;cy=float(vals[1]-.5)*.9
        a=.12+float(vals[2])*.32;b=.10+float(vals[3])*.28;ang=float(vals[4])*math.pi
        xr=(xx-cx)*math.cos(ang)+(yy-cy)*math.sin(ang)
        yr=-(xx-cx)*math.sin(ang)+(yy-cy)*math.cos(ang)
        amp=float(torch.rand((),generator=g))
        image+=amp*((xr/a).square()+(yr/b).square()<1)
    image=image.clamp_min(0);image=image/(image.max()+1e-8)
    phase=.08*torch.sin(math.pi*xx)*torch.cos(math.pi*yy)
    img=torch.polar(image,phase).to(torch.complex64)
    shift=0.0 if scanner=='A' else .57
    sens=[]
    for c in range(coils):
        ang=2*math.pi*c/coils+shift
        amp=1+.22*torch.cos(math.pi*xx*math.cos(ang)+math.pi*yy*math.sin(ang)+shift)
        ph=.35*(xx*math.cos(ang)+yy*math.sin(ang))
        if scanner=='B':ph=ph+.65*torch.sin(math.pi*yy+ang)
        sens.append(torch.polar(amp,ph))
    sens=torch.stack(sens).to(torch.complex64)
    sens=sens/(sens.abs().square().sum(dim=0,keepdim=True).sqrt()+1e-8)
    line=torch.zeros(size,dtype=torch.float32);line[::accel]=1
    lo=size//2-center//2;line[lo:lo+center]=1
    mask=line.view(1,1,size).to(torch.complex64)
    y=forward_op(img,sens,mask)
    return {'y':y,'sens':sens,'mask':mask,'target':image,'seed':int(seed),'scanner':scanner}

class ResidualDenoiser(nn.Module):
    def __init__(self,hidden:int=8):
        super().__init__();self.conv1=nn.Conv2d(2,hidden,3,padding=1);self.conv2=nn.Conv2d(hidden,hidden,3,padding=1);self.conv3=nn.Conv2d(hidden,2,3,padding=1)
    def forward(self,x:torch.Tensor,angle:torch.Tensor|None=None):
        z=torch.stack((x.real,x.imag),dim=0).unsqueeze(0)
        h=F.relu(self.conv1(z))
        if angle is not None:
            b,c,hg,wg=h.shape;pair=h.reshape(b,c//2,2,hg,wg);co=torch.cos(angle);si=torch.sin(angle)
            a=co*pair[:,:,0]-si*pair[:,:,1];q=si*pair[:,:,0]+co*pair[:,:,1]
            h=torch.stack((a,q),dim=2).reshape(b,c,hg,wg)
        h=F.relu(self.conv2(h));r=self.conv3(h)[0]
        return x+torch.complex(r[0],r[1])

class MoDLNet(nn.Module):
    """K-step MoDL with CG normal-equation solves and controlled sharing/view mode."""
    def __init__(self,mode:str='tied',steps:int=3,cg_steps:int=5,hidden:int=8):
        super().__init__();assert mode in {'tied','step_lambda','mirror','independent'}
        self.mode=mode;self.steps=steps;self.cg_steps=cg_steps;self.hidden=hidden
        if mode=='independent':self.denoisers=nn.ModuleList([ResidualDenoiser(hidden) for _ in range(steps)])
        else:self.denoiser=ResidualDenoiser(hidden)
        self.raw_lambda=nn.Parameter(torch.full((steps if mode=='step_lambda' else 1,),-2.5))
        if mode=='mirror':self.angles=nn.Parameter(torch.zeros(steps))
    def lam(self,k):return F.softplus(self.raw_lambda[k if self.mode=='step_lambda' else 0])+1e-4
    def denoise(self,x,k):
        if self.mode=='independent':return self.denoisers[k](x)
        return self.denoiser(x,self.angles[k] if self.mode=='mirror' else None)
    def solve(self,rhs,lam,sens,mask):
        def op(x):return normal_op(x,sens,mask)+lam*x
        x=torch.zeros_like(rhs);r=rhs-op(x);p=r;rr=(r.conj()*r).real.sum()+1e-12;rr0=rr
        for _ in range(self.cg_steps):
            ap=op(p);den=(p.conj()*ap).real.sum()+1e-12;alpha=rr/den;x=x+alpha*p;r=r-alpha*ap
            rr_new=(r.conj()*r).real.sum()+1e-12;beta=rr_new/rr;p=r+beta*p;rr=rr_new
        rel=float(torch.sqrt(rr/(rr0+1e-12)).detach())
        return x,rel
    def forward(self,case):
        y,sens,mask=case['y'],case['sens'],case['mask'];atb=adjoint_op(y,sens,mask);x=atb;cg=[]
        for k in range(self.steps):
            z=self.denoise(x,k);lam=self.lam(k);x,res=self.solve(atb+lam*z,lam,sens,mask);cg.append(res)
        return x,cg

def rel_measurement_residual(x,case):
    r=forward_op(x,case['sens'],case['mask'])-case['y']
    return float(r.norm()/(case['y'].norm()+1e-12))
