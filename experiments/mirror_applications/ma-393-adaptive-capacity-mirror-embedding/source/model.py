"""Frequency-banded embedding models for MA-393."""
from __future__ import annotations

import torch

VOCAB=384
DIM=16
BAND_SIZES=(64,128,192)
BAND_WIDTHS=(16,8,4)
BAND_MASSES=(0.70,0.22,0.08)
CLASSES=16
METHODS=('full','adaptive','byte_control','mirror')


def band_ids()->torch.Tensor:
    return torch.cat([torch.full((n,),i,dtype=torch.long) for i,n in enumerate(BAND_SIZES)])


def projection(seed:int,band:int,width:int)->torch.Tensor:
    g=torch.Generator().manual_seed(seed+1009*(band+1)+97*width)
    return torch.randn(DIM,width,generator=g)/width**0.5


def rotate_first_pair(x:torch.Tensor,angle:torch.Tensor)->torch.Tensor:
    c=torch.cos(angle);s=torch.sin(angle)
    y=x.clone()
    y[...,0]=c*x[...,0]-s*x[...,1]
    y[...,1]=s*x[...,0]+c*x[...,1]
    return y


class AdaptiveBank(torch.nn.Module):
    def __init__(self,method:str,init_seed:int,basis_seed:int,decoder:torch.Tensor):
        super().__init__()
        if method not in METHODS:raise ValueError(method)
        self.method=method;self.basis_seed=int(basis_seed)
        self.register_buffer('decoder',decoder.detach().clone().float())
        self.register_buffer('band_id',band_ids())
        for b,k in enumerate(BAND_WIDTHS):
            width=k+1 if method=='byte_control' else k
            self.register_buffer(f'projection_{b}',projection(self.basis_seed,b,width))
        torch.manual_seed(init_seed)
        if method=='full':
            self.embedding=torch.nn.Parameter(0.1*torch.randn(VOCAB,DIM))
        else:
            self.codes=torch.nn.Parameter(0.1*torch.randn(VOCAB,max(BAND_WIDTHS)+1))
            if method=='mirror':self.angle=torch.nn.Parameter(2*torch.pi*torch.rand(VOCAB))

    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        if self.method=='full':return self.embedding[token_ids]
        out=torch.empty(token_ids.shape+(DIM,),dtype=self.decoder.dtype,device=token_ids.device)
        for b,k in enumerate(BAND_WIDTHS):
            select=self.band_id[token_ids]==b
            if not bool(select.any()):continue
            ids=token_ids[select];width=k+1 if self.method=='byte_control' else k
            y=self.codes[ids,:width]@getattr(self,f'projection_{b}').T
            if self.method=='mirror':y=rotate_first_pair(y,self.angle[ids])
            out[select]=y
        return out

    def payload_arrays(self)->dict[str,object]:
        out:dict[str,object]={'method':self.method,'vocabulary_size':VOCAB,'embedding_dim':DIM,
            'band_sizes':torch.tensor(BAND_SIZES).numpy(),'band_widths':torch.tensor(BAND_WIDTHS).numpy(),
            'band_masses':torch.tensor(BAND_MASSES).numpy(),'decoder':self.decoder.detach().cpu().numpy().astype('<f4')}
        if self.method=='full':out['embedding']=self.embedding.detach().cpu().numpy().astype('<f4')
        else:
            out['basis_seed']=self.basis_seed
            for b,n in enumerate(BAND_SIZES):
                k=BAND_WIDTHS[b]+1 if self.method=='byte_control' else BAND_WIDTHS[b]
                start=sum(BAND_SIZES[:b])
                out[f'code_band_{b}']=self.codes[start:start+n,:k].detach().cpu().numpy().astype('<f4')
            if self.method=='mirror':out['angle']=self.angle.detach().cpu().numpy().astype('<f4')
        return out


def from_payload(payload:dict[str,object])->AdaptiveBank:
    method=str(payload['method']);basis_seed=int(payload.get('basis_seed',0))
    decoder=torch.as_tensor(payload['decoder'],dtype=torch.float32)
    bank=AdaptiveBank(method,0,basis_seed,decoder)
    with torch.no_grad():
        if method=='full':bank.embedding.copy_(torch.as_tensor(payload['embedding'],dtype=torch.float32))
        else:
            for b,n in enumerate(BAND_SIZES):
                start=sum(BAND_SIZES[:b]);value=torch.as_tensor(payload[f'code_band_{b}'],dtype=torch.float32)
                bank.codes[start:start+n,:value.shape[1]].copy_(value)
            if method=='mirror':bank.angle.copy_(torch.as_tensor(payload['angle'],dtype=torch.float32))
    return bank.eval()


def compute_proxy(method:str)->dict[str,float|int]:
    widths=[k+1 if method=='byte_control' else k for k in BAND_WIDTHS]
    avg_projection=sum(m*w for m,w in zip(BAND_MASSES,widths))*DIM
    return {'frequency_weighted_projection_MAC_per_token':0 if method=='full' else avg_projection,
            'mirror_rotation_macs_per_token':4.0 if method=='mirror' else 0.0,
            'mirror_trig_ops_per_token':2 if method=='mirror' else 0,
            'fixed_decoder_MAC_per_token':DIM*CLASSES}
