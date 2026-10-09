"""Domain-conditioned shared token embedding views for MA-392."""
from __future__ import annotations

import torch

VOCAB=128
DOMAINS=8
DIM=16
CLASSES=16
METHODS=("oracle","shared","film","dense","mirror")


class DomainBank(torch.nn.Module):
    def __init__(self,method:str,base_table:torch.Tensor,decoder:torch.Tensor,embedding_table:torch.Tensor|None=None):
        super().__init__()
        if method not in METHODS: raise ValueError(method)
        self.method=method
        self.register_buffer('base_table',base_table.detach().clone().float())
        self.register_buffer('decoder',decoder.detach().clone().float())
        if method=='oracle':
            if embedding_table is None: raise ValueError('oracle requires full pair table')
            self.register_buffer('embedding_table',embedding_table.detach().clone().float())
        elif method=='film':
            self.scale=torch.nn.Parameter(torch.ones(DOMAINS,DIM))
            self.shift=torch.nn.Parameter(torch.zeros(DOMAINS,DIM))
        elif method=='dense':
            eye=torch.eye(DIM).expand(DOMAINS,DIM,DIM).clone()
            self.transform=torch.nn.Parameter(eye)
        elif method=='mirror':
            self.angle=torch.nn.Parameter(torch.zeros(DOMAINS,DIM//2))

    def forward(self,token_ids:torch.Tensor,domain_ids:torch.Tensor)->torch.Tensor:
        if self.method=='oracle': return self.embedding_table[domain_ids,token_ids]
        x=self.base_table[token_ids]
        if self.method=='shared': return x
        if self.method=='film':
            return x*self.scale[domain_ids]+self.shift[domain_ids]
        if self.method=='dense':
            return torch.bmm(x.unsqueeze(1),self.transform[domain_ids].transpose(1,2)).squeeze(1)
        if self.method=='mirror':
            a=self.angle[domain_ids]
            c=torch.cos(a); s=torch.sin(a)
            x0=x[:,0::2]; x1=x[:,1::2]
            y=torch.empty_like(x)
            y[:,0::2]=c*x0-s*x1
            y[:,1::2]=s*x0+c*x1
            return y
        raise AssertionError(self.method)

    def payload_arrays(self)->dict[str,object]:
        out:dict[str,object]={'method':self.method,'vocabulary_size':VOCAB,'domain_count':DOMAINS,
            'embedding_dim':DIM,'decoder':self.decoder.detach().cpu().numpy().astype('<f4')}
        if self.method=='oracle':
            out['embedding_table']=self.embedding_table.detach().cpu().numpy().astype('<f4')
        else:
            out['base_table']=self.base_table.detach().cpu().numpy().astype('<f4')
        for name,param in self.named_parameters():out[f'param_{name}']=param.detach().cpu().numpy().astype('<f4')
        return out


def from_payload(payload:dict[str,object])->DomainBank:
    method=str(payload['method'])
    base=torch.as_tensor(payload.get('base_table',torch.zeros(VOCAB,DIM)),dtype=torch.float32)
    decoder=torch.as_tensor(payload['decoder'],dtype=torch.float32)
    full=torch.as_tensor(payload['embedding_table'],dtype=torch.float32) if 'embedding_table' in payload else None
    bank=DomainBank(method,base,decoder,full)
    with torch.no_grad():
        for name,param in bank.named_parameters():
            key=f'param_{name}'
            if key not in payload: raise ValueError(f'missing parameter {key}')
            param.copy_(torch.as_tensor(payload[key],dtype=param.dtype))
    return bank.eval()


def compute_proxy(method:str)->dict[str,int]:
    return {'token_embedding_lookups_per_pair':1 if method!='oracle' else 1,
            'domain_transform_MAC_per_pair':{'oracle':0,'shared':0,'film':2*DIM,'dense':DIM*DIM,'mirror':2*DIM}[method],
            'domain_transform_adds_per_pair':{'oracle':0,'shared':0,'film':DIM,'dense':0,'mirror':DIM}[method],
            'mirror_trig_ops_per_pair':2*(DIM//2) if method=='mirror' else 0,
            'fixed_decoder_MAC_per_pair':DIM*CLASSES}
