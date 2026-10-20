"""MA-1170 minimal Mirror virtual-graph, matched affine and sparse controls.
All logical edge tensors are generated from a single 64-value learned graph program.
"""
from __future__ import annotations
import hashlib
import json
import torch
from torch import nn
import torch.nn.functional as F

N_MELS=64
N_LATENT=16
TOP_K=6


def topk_relu(t:torch.Tensor, k:int=TOP_K)->torch.Tensor:
    positive=F.relu(t)
    keep=torch.topk(positive,k,dim=-1).indices
    out=torch.zeros_like(positive)
    return out.scatter(-1,keep,positive.gather(-1,keep))


def fixed_program(seed:int=1170):
    gen=torch.Generator().manual_seed(seed)
    maps=[torch.arange(16)]
    signs=[torch.ones(16)]
    for _ in range(3):
        maps.append(torch.randperm(16,generator=gen))
        signs.append(2*torch.randint(0,2,(16,),generator=gen,dtype=torch.int32).float()-1)
    perms=torch.stack(maps)
    signs=torch.stack(signs)
    v=torch.randn(16,generator=gen); v=v/v.norm()
    H=torch.eye(16)-2*torch.outer(v,v)
    page_keys=torch.tensor([[1.0,0.0],[0.0,1.0],[-1.0,0.0],[0.0,-1.0]],dtype=torch.float32)
    return perms,signs,H,page_keys


class GraphProgram(nn.Module):
    """Generated matrices use 16 core+16 gate+32 address learned floats =64.
    4 pages and Householder remain deterministic; nonpersistent buffers regenerated at load.
    """
    def __init__(self,kind:str):
        super().__init__()
        assert kind in ['mirror','affine']
        self.kind=kind
        self.core=nn.Parameter(0.2*torch.randn(16))
        self.gates=nn.Parameter(torch.full((16,),-1.0))
        self.address=nn.Parameter(0.2*torch.randn(16,2))
        perm,sgn,H,keys=fixed_program()
        self.register_buffer('perms',perm,persistent=False)
        self.register_buffer('signs',sgn,persistent=False)
        self.register_buffer('householder',H,persistent=False)
        self.register_buffer('page_keys',keys,persistent=False)

    def weight_matrices(self,disabled:int=0):
        pages=self.core[self.perms]*self.signs
        logits=self.address@self.page_keys.T*3.0
        mix=torch.softmax(logits,dim=-1)
        v=mix@pages
        if self.kind=='mirror':
            # H is symmetric and H.T@H=I. A nonlinear but smooth weight-space view.
            a=v@self.householder.T
            v=torch.tanh(3.0*a)@self.householder / 3.0
        weight=v.reshape(4,4,4,4)
        gate=torch.sigmoid(self.gates)
        if disabled:
            # Only for post-training ablation; do not modify model parameters.
            sel=torch.argsort(gate)[:disabled]
            mask=torch.ones_like(gate).scatter(0,sel,0.0)
            gate=gate*mask
        return weight*gate.reshape(4,4,1,1)

    def forward(self,z:torch.Tensor,disabled:int=0):
        weights=self.weight_matrices(disabled=disabled)
        state=z.reshape(-1,4,4)
        for _ in range(2):
            msg=torch.einsum('dsij,bsj->bdi',weights,state)
            state=F.relu(state+0.5*msg)
        return state.reshape(-1,16)


class AudioAE(nn.Module):
    def __init__(self,kind:str):
        super().__init__()
        assert kind in ['mirror','affine','sparse','native']
        self.kind=kind
        # Exactly 1024+16+1024+16 = 2080 learned coefficients in the shared codec.
        self.encoder=nn.Linear(N_MELS,N_LATENT,bias=True)
        self.latent_bias=nn.Parameter(torch.zeros(N_LATENT))
        self.decoder=nn.Linear(N_LATENT,N_MELS,bias=False)
        if kind in ['mirror','affine']:
            self.graph=GraphProgram(kind)
        elif kind=='native':
            self.hidden=nn.Linear(N_LATENT,N_LATENT,bias=True)
        n=sum(p.numel() for p in self.parameters())
        assert n=={'mirror':2144,'affine':2144,'sparse':2080,'native':2352}[kind],(kind,n)

    def encode(self,x):
        return topk_relu(self.encoder(x)+self.latent_bias)

    def decode(self,z:torch.Tensor,masked_edges:int=0):
        if self.kind in ('mirror','affine'):
            z=self.graph(z,disabled=masked_edges)
        elif self.kind=='native':
            z=F.relu(z+0.25*self.hidden(z))
        elif masked_edges:
            raise ValueError('edge ablation only supported for graph models')
        return self.decoder(z)

    def forward(self,x:torch.Tensor,masked_edges:int=0):
        return self.decode(self.encode(x),masked_edges=masked_edges)


def program_sha256()->str:
    ps= fixed_program()
    h=hashlib.sha256()
    for t in ps:
        h.update(t.numpy().tobytes())
    return h.hexdigest()
