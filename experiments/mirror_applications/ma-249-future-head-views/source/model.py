from __future__ import annotations
import io,json,math
import torch
from torch import nn

METHODS=('mtp','tied','scalar_gate','lowrank','mirror')

def givens(x,angles):
    b,h,d=x.shape;p=x.reshape(b,h,d//2,2);a=angles[None];c,s=torch.cos(a),torch.sin(a);u,v=p[...,0],p[...,1]
    return torch.stack((c*u-s*v,s*u+c*v),-1).reshape(b,h,d)

class HeadStudent(nn.Module):
    def __init__(self,method,feature_map,offsets=4,vocab=12,d=16,rank=1):
        super().__init__();self.method=method;self.offsets=offsets;self.vocab=vocab;self.d=d;self.rank=rank
        self.register_buffer('feature_map',feature_map.detach().clone())
        if method=='mtp':
            self.weight=nn.Parameter(torch.randn(offsets,vocab,d)*.08);self.bias=nn.Parameter(torch.zeros(offsets,vocab))
        elif method in ('tied','scalar_gate'):
            self.weight=nn.Parameter(torch.randn(vocab,d)*.08);self.bias=nn.Parameter(torch.zeros(vocab))
            if method=='scalar_gate':self.gate=nn.Parameter(torch.ones(offsets))
        elif method=='lowrank':
            self.weight=nn.Parameter(torch.randn(vocab,d)*.08);self.bias=nn.Parameter(torch.zeros(vocab))
            self.a=nn.Parameter(torch.zeros(offsets,vocab,rank));self.b=nn.Parameter(torch.randn(offsets,rank,d)*.02)
        elif method=='mirror':
            self.weight=nn.Parameter(torch.randn(vocab,d)*.08);self.bias=nn.Parameter(torch.zeros(vocab));self.angles=nn.Parameter(torch.zeros(offsets,d//2))
        else:raise ValueError(method)
    def forward(self,x):
        h=x@self.feature_map
        if self.method=='mtp':return torch.einsum('bd,hvd->bhv',h,self.weight)+self.bias[None]
        if self.method=='mirror':h=givens(h[:,None,:].expand(-1,self.offsets,-1),self.angles)
        else:h=h[:,None,:].expand(-1,self.offsets,-1)
        logits=torch.einsum('bhd,vd->bhv',h,self.weight)+self.bias
        if self.method=='scalar_gate':logits=logits*self.gate[None,:,None]
        if self.method=='lowrank':logits=logits+torch.einsum('bhd,hrd,hvr->bhv',h,self.b,self.a)
        return logits
    def serialize(self):
        cfg={'method':self.method,'offsets':self.offsets,'vocab':self.vocab,'d':self.d,'rank':self.rank}
        buf=io.BytesIO();torch.save({'state_dict':self.state_dict(),'config':cfg},buf)
        return buf.getvalue()+json.dumps(cfg,sort_keys=True).encode()
    def serialized_payload_bytes(self):return len(self.serialize())

def compute_proxy(method,examples,offsets=4,vocab=12,d=16):
    mac=offsets*vocab*d
    if method=='mirror':mac+=offsets*d
    if method=='lowrank':mac+=offsets*(vocab+d)
    return int(mac*examples*3)
