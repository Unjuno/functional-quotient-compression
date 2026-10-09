"""Collision-heavy product-address embedding bank for MA-397."""
from __future__ import annotations
import torch

VOCAB=2048;BUCKETS=32;DIM=16;CLASSES=16
METHODS=('full','product','coeff','mirror')


def product_indices(tokens:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    t=tokens.to(torch.long)
    return torch.remainder(t,BUCKETS),torch.remainder(torch.div(t,BUCKETS,rounding_mode='floor'),BUCKETS)


class ProductBank(torch.nn.Module):
    def __init__(self,method:str,seed:int,table0:torch.Tensor,table1:torch.Tensor,decoder:torch.Tensor):
        super().__init__()
        if method not in METHODS:raise ValueError(method)
        self.method=method
        self.register_buffer('table0',table0.detach().clone().float())
        self.register_buffer('table1',table1.detach().clone().float())
        self.register_buffer('decoder',decoder.detach().clone().float())
        torch.manual_seed(seed)
        if method=='full':self.embedding=torch.nn.Parameter(0.1*torch.randn(VOCAB,DIM))
        elif method=='coeff':self.coeff=torch.nn.Parameter(0.1*torch.randn(VOCAB,2))
        elif method=='mirror':self.angle=torch.nn.Parameter(2*torch.pi*torch.rand(VOCAB))

    def forward(self,tokens:torch.Tensor)->torch.Tensor:
        if self.method=='full':return self.embedding[tokens]
        a,b=product_indices(tokens);x=self.table0[a];y=self.table1[b]
        if self.method=='product':return x+y
        if self.method=='coeff':
            c=self.coeff[tokens];return c[:,0:1]*x+c[:,1:2]*y
        if self.method=='mirror':
            th=self.angle[tokens];return torch.cos(th)[:,None]*x+torch.sin(th)[:,None]*y
        raise AssertionError(self.method)

    def payload_arrays(self)->dict[str,object]:
        out={'method':self.method,'vocabulary_size':VOCAB,'bucket_count':BUCKETS,'embedding_dim':DIM,
             'decoder':self.decoder.detach().cpu().numpy().astype('<f4')}
        if self.method=='full':out['embedding']=self.embedding.detach().cpu().numpy().astype('<f4')
        else:
            out['table0']=self.table0.detach().cpu().numpy().astype('<f4');out['table1']=self.table1.detach().cpu().numpy().astype('<f4')
            for name,p in self.named_parameters():out['param_'+name]=p.detach().cpu().numpy().astype('<f4')
        return out


def from_payload(payload:dict[str,object])->ProductBank:
    method=str(payload['method']);tab0=torch.as_tensor(payload.get('table0',torch.zeros(BUCKETS,DIM)),dtype=torch.float32)
    tab1=torch.as_tensor(payload.get('table1',torch.zeros(BUCKETS,DIM)),dtype=torch.float32)
    decoder=torch.as_tensor(payload['decoder'],dtype=torch.float32);bank=ProductBank(method,0,tab0,tab1,decoder)
    with torch.no_grad():
        for name,p in bank.named_parameters():
            key='param_'+name if method!='full' else name
            if method=='full':value=payload['embedding']
            else:
                if key not in payload:raise ValueError('missing '+key)
                value=payload[key]
            p.copy_(torch.as_tensor(value,dtype=p.dtype))
    return bank.eval()


def compute_proxy(method:str)->dict[str,int]:
    return {'component_row_lookups':0 if method=='full' else 2,
            'composition_MAC_per_token':{'full':0,'product':DIM,'coeff':2*DIM,'mirror':2*DIM}[method],
            'mirror_trig_ops_per_token':2 if method=='mirror' else 0,
            'decoder_MAC_per_token':DIM*CLASSES}
