"""ALBERT-style factorized token embeddings with domain views."""
from __future__ import annotations
import torch

VOCAB=128;DOMAINS=8;LATENT=8;DIM=16;CLASSES=16
METHODS=('oracle','shared','film','dense','mirror')


class FactorizedDomainBank(torch.nn.Module):
    def __init__(self,method:str,base:torch.Tensor,projection:torch.Tensor,decoder:torch.Tensor,oracle:torch.Tensor|None=None):
        super().__init__()
        if method not in METHODS:raise ValueError(method)
        self.method=method
        self.register_buffer('base',base.detach().clone().float())
        self.register_buffer('projection',projection.detach().clone().float())
        self.register_buffer('decoder',decoder.detach().clone().float())
        if method=='oracle':
            if oracle is None:raise ValueError('oracle requires output table')
            self.register_buffer('oracle',oracle.detach().clone().float())
        elif method=='film':
            self.scale=torch.nn.Parameter(torch.ones(DOMAINS,DIM));self.shift=torch.nn.Parameter(torch.zeros(DOMAINS,DIM))
        elif method=='dense':
            self.transform=torch.nn.Parameter(torch.eye(LATENT).expand(DOMAINS,LATENT,LATENT).clone())
        elif method=='mirror':
            self.angle=torch.nn.Parameter(torch.zeros(DOMAINS,LATENT//2))

    def forward(self,tokens:torch.Tensor,domains:torch.Tensor)->torch.Tensor:
        if self.method=='oracle':return self.oracle[domains,tokens]
        z=self.base[tokens]
        if self.method=='shared':y=z@self.projection
        elif self.method=='film':y=(z@self.projection)*self.scale[domains]+self.shift[domains]
        elif self.method=='dense':
            latent=torch.bmm(z.unsqueeze(1),self.transform[domains].transpose(1,2)).squeeze(1)
            y=latent@self.projection
        elif self.method=='mirror':
            a=self.angle[domains];c=torch.cos(a);s=torch.sin(a)
            z0=z[:,0::2];z1=z[:,1::2];rot=torch.empty_like(z)
            rot[:,0::2]=c*z0-s*z1;rot[:,1::2]=s*z0+c*z1
            y=rot@self.projection
        else:raise AssertionError(self.method)
        return y

    def payload_arrays(self)->dict[str,object]:
        out={'method':self.method,'vocabulary_size':VOCAB,'domain_count':DOMAINS,'latent_dim':LATENT,
             'output_dim':DIM,'decoder':self.decoder.detach().cpu().numpy().astype('<f4')}
        if self.method=='oracle':out['oracle']=self.oracle.detach().cpu().numpy().astype('<f4')
        else:
            out['base']=self.base.detach().cpu().numpy().astype('<f4')
            out['projection']=self.projection.detach().cpu().numpy().astype('<f4')
        for name,param in self.named_parameters():out['param_'+name]=param.detach().cpu().numpy().astype('<f4')
        return out


def from_payload(payload:dict[str,object])->FactorizedDomainBank:
    method=str(payload['method'])
    base=torch.as_tensor(payload.get('base',torch.zeros(VOCAB,LATENT)),dtype=torch.float32)
    projection=torch.as_tensor(payload.get('projection',torch.zeros(LATENT,DIM)),dtype=torch.float32)
    decoder=torch.as_tensor(payload['decoder'],dtype=torch.float32)
    oracle=torch.as_tensor(payload['oracle'],dtype=torch.float32) if 'oracle' in payload else None
    bank=FactorizedDomainBank(method,base,projection,decoder,oracle)
    with torch.no_grad():
        for name,param in bank.named_parameters():
            key='param_'+name
            if key not in payload:raise ValueError('Missing '+key)
            param.copy_(torch.as_tensor(payload[key],dtype=param.dtype))
    return bank.eval()


def compute_proxy(method:str)->dict[str,int]:
    return {'latent_projection_MAC':LATENT*DIM if method!='oracle' else 0,
            'domain_transform_MAC':{'oracle':0,'shared':0,'film':2*DIM,'dense':LATENT*LATENT,'mirror':2*LATENT}[method],
            'domain_transform_adds':{'oracle':0,'shared':0,'film':DIM,'dense':0,'mirror':LATENT}[method],
            'mirror_trig_ops':LATENT if method=='mirror' else 0,
            'decoder_MAC':DIM*CLASSES}
