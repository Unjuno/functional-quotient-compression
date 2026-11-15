"""Small PEFT families attached to the frozen SAM final image embedding."""
import torch
from torch import nn
from torch.nn import functional as F

DOMAINS=('texture_contrast','low_contrast_camouflage','partial_occlusion')

class Bottleneck(nn.Module):
    def __init__(self,channels=256,rank=8,nonlinear=True):
        super().__init__();self.down=nn.Conv2d(channels,rank,1);self.up=nn.Conv2d(rank,channels,1);self.nonlinear=nonlinear
        nn.init.kaiming_uniform_(self.down.weight,a=5**.5);nn.init.zeros_(self.down.bias);nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def latent(self,x):
        z=self.down(x)
        return F.relu(z) if self.nonlinear else z
    def project(self,z):return self.up(z)
    def forward(self,x):return x+self.project(self.latent(x))

class IndependentAdapters(nn.Module):
    def __init__(self):super().__init__();self.adapters=nn.ModuleList([Bottleneck() for _ in DOMAINS])
    def apply(self,x,task):return self.adapters[task](x)
    def active(self,task):return self.adapters[task]

class SharedAdapter(nn.Module):
    def __init__(self):super().__init__();self.adapter=Bottleneck()
    def apply(self,x,task):return self.adapter(x)
    def active(self,task):return self.adapter

class MirrorGivens(nn.Module):
    def __init__(self):super().__init__();self.adapter=Bottleneck();self.angles=nn.Parameter(torch.zeros(len(DOMAINS),2))
    def apply(self,x,task):
        z=self.adapter.latent(x);a=self.angles[task]
        c0,s0=torch.cos(a[0]),torch.sin(a[0]);c1,s1=torch.cos(a[1]),torch.sin(a[1]);y=z.clone()
        x0,x1=z[:,0].clone(),z[:,1].clone();y[:,0]=c0*x0-s0*x1;y[:,1]=s0*x0+c0*x1
        x2,x3=z[:,2].clone(),z[:,3].clone();y[:,2]=c1*x2-s1*x3;y[:,3]=s1*x2+c1*x3
        return x+self.adapter.project(y)
    def active(self,task):return self.adapter

class FiLMControl(nn.Module):
    def __init__(self):super().__init__();self.adapter=Bottleneck();self.scale=nn.Parameter(torch.zeros(len(DOMAINS),8));self.shift=nn.Parameter(torch.zeros(len(DOMAINS),8))
    def apply(self,x,task):
        z=self.adapter.latent(x);z=z*(1+self.scale[task][None,:,None,None])+self.shift[task][None,:,None,None]
        return x+self.adapter.project(z)
    def active(self,task):return self.adapter

class MoPEFTLike(nn.Module):
    """Feature-space mixture: Adapter, linear low-rank branch, prompt-vector residual, task gate."""
    def __init__(self):
        super().__init__();self.adapter=Bottleneck();self.lora=Bottleneck(nonlinear=False)
        self.prefix=nn.Parameter(torch.zeros(len(DOMAINS),256));self.gate_logits=nn.Parameter(torch.zeros(len(DOMAINS),3))
    def apply(self,x,task):
        w=F.softmax(self.gate_logits[task],dim=-1)
        da=self.adapter.project(self.adapter.latent(x));dl=self.lora.project(self.lora.latent(x));dp=self.prefix[task][None,:,None,None]
        return x+w[0]*da+w[1]*dl+w[2]*dp
    def active(self,task):return self

FACTORIES={'independent_adapter':IndependentAdapters,'hard_shared_adapter':SharedAdapter,'mirror_givens':MirrorGivens,'film_control':FiLMControl,'mopeft_like':MoPEFTLike}
