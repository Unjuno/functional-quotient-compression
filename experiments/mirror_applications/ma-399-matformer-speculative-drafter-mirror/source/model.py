"""Nested supermodel draft roles and Mirror correction for MA-399."""
from __future__ import annotations
import torch

INPUT=8;WIDTH=16;VOCAB=16;METHODS=('full_only','nested8','nested12','mirror','film','independent8')


def rotate8(h:torch.Tensor,angle:torch.Tensor)->torch.Tensor:
    c=torch.cos(angle);s=torch.sin(angle);y=h.clone()
    for j in range(4):
        x=h[...,2*j];z=h[...,2*j+1]
        y[...,2*j]=c[j]*x-s[j]*z;y[...,2*j+1]=s[j]*x+c[j]*z
    return y


class SpecSystem(torch.nn.Module):
    def __init__(self,method:str,w1:torch.Tensor,b1:torch.Tensor,w2:torch.Tensor,b2:torch.Tensor):
        super().__init__()
        if method not in METHODS:raise ValueError(method)
        self.method=method
        self.register_buffer('w1',w1.detach().clone().float());self.register_buffer('b1',b1.detach().clone().float())
        self.register_buffer('w2',w2.detach().clone().float());self.register_buffer('b2',b2.detach().clone().float())
        if method=='mirror':self.angle=torch.nn.Parameter(torch.zeros(4))
        elif method=='film':
            self.scale=torch.nn.Parameter(torch.ones(8));self.shift=torch.nn.Parameter(torch.zeros(8))
        elif method=='independent8':
            self.d1=torch.nn.Parameter(w1[:,:8].detach().clone())
            self.db1=torch.nn.Parameter(b1[:8].detach().clone())
            self.d2=torch.nn.Parameter(w2[:8].detach().clone())
            self.db2=torch.nn.Parameter(b2.detach().clone())

    def verifier_logits(self,x:torch.Tensor)->torch.Tensor:
        h=torch.tanh(x@self.w1+self.b1)
        return h@self.w2+self.b2

    def draft_logits(self,x:torch.Tensor)->torch.Tensor:
        if self.method=='full_only':return self.verifier_logits(x)
        if self.method=='independent8':return torch.tanh(x@self.d1+self.db1)@self.d2+self.db2
        width=8 if self.method in ('nested8','mirror','film') else 12
        h=torch.tanh(x@self.w1[:,:width]+self.b1[:width])
        if self.method=='mirror':h=rotate8(h,self.angle)
        elif self.method=='film':h=h*self.scale+self.shift
        return h@self.w2[:width]+self.b2

    def payload_arrays(self)->dict[str,object]:
        out={'method':self.method,'input_dim':INPUT,'full_width':WIDTH,'vocabulary_size':VOCAB,
             'w1':self.w1.detach().cpu().numpy().astype('<f4'),'b1':self.b1.detach().cpu().numpy().astype('<f4'),
             'w2':self.w2.detach().cpu().numpy().astype('<f4'),'b2':self.b2.detach().cpu().numpy().astype('<f4')}
        for name,p in self.named_parameters():out['param_'+name]=p.detach().cpu().numpy().astype('<f4')
        return out


def from_payload(payload:dict[str,object])->SpecSystem:
    tensors={k:torch.as_tensor(payload[k],dtype=torch.float32) for k in ('w1','b1','w2','b2')}
    sys=SpecSystem(str(payload['method']),**tensors)
    with torch.no_grad():
        for name,p in sys.named_parameters():
            key='param_'+name
            if key not in payload:raise ValueError('missing '+key)
            p.copy_(torch.as_tensor(payload[key],dtype=p.dtype))
    return sys.eval()


def speculative_distribution(q:torch.Tensor,p:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    """Expected output distribution and per-row acceptance probability."""
    accepted=torch.minimum(q,p)
    a=accepted.sum(dim=-1,keepdim=True)
    residual=(p-q).clamp_min(0)
    residual_mass=residual.sum(dim=-1,keepdim=True)
    correction=torch.where(residual_mass>0,residual/residual_mass.clamp_min(1e-30),p)
    output=accepted+(1-a)*correction
    return output,a.squeeze(-1)


def compute_proxy(method:str)->dict[str,int]:
    width={'full_only':16,'nested8':8,'nested12':12,'mirror':8,'film':8,'independent8':8}[method]
    return {'draft_MAC_per_context':INPUT*width+width*VOCAB,
            'verifier_MAC_per_context':INPUT*WIDTH+WIDTH*VOCAB,
            'mirror_rotation_multiplies':24 if method=='mirror' else 0,
            'mirror_trig_ops':8 if method=='mirror' else 0,
            'film_affine_ops':16 if method=='film' else 0}
