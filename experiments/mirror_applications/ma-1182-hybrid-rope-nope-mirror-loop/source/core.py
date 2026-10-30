"""MA-1182 frozen model/data definitions; no access to fresh during training."""
from __future__ import annotations
import hashlib,json,io,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PROTOCOL=json.loads((Path(__file__).resolve().parents[1]/'PROTOCOL.json').read_text())
P=PROTOCOL['model']; D=P['width']; H=P['head_width']; HEADS=P['heads']; VOCAB=P['vocab']
POSITIONS=P['positions']; ARMS=P['arms']; TASKS=list(PROTOCOL['tasks'])

def reflect(x,v):
    u=v/v.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    return x-2*(x*u).sum(-1,keepdim=True)*u

def rank1_trans(x,v,beta):
    u=v/v.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    return x+beta*(x*u).sum(-1,keepdim=True)*u

def rotary(x):
    """x of shape batch x positions x D_head, operates on adjacent coordinate pairs."""
    l=x.shape[-2]
    t=torch.arange(l,dtype=x.dtype,device=x.device)
    inv=(10000.0)**(-torch.arange(0,H,2,device=x.device,dtype=x.dtype)/H)
    ang=t[:,None]*inv[None,:]
    s,c=torch.sin(ang),torch.cos(ang)
    a,b=x[...,::2],x[...,1::2]
    return torch.stack([c*a-s*b,s*a+c*b],dim=-1).flatten(-2)

def make_data(task,seed,n,split,content_length=None):
    """No file- or model-dependent RNG. A query is the final token."""
    if content_length is None: content_length=P['train_sequence_content_length']
    assert task in TASKS and split in (1,2,3,4) and content_length>=5
    rng=np.random.default_rng(4309827+seed*98191+TASKS.index(task)*991211+split*1237+content_length*173)
    yy=rng.integers(0,2,size=n,dtype=np.int64)
    xx=np.empty((n,content_length+1),dtype=np.int64)
    for i,y in enumerate(yy):
        key=int(rng.integers(0,4))
        if task=='set_membership':
            if y:
                keys=rng.integers(0,4,size=content_length)
                keys[int(rng.integers(0,content_length))]=key
            else:
                ks=np.array([k for k in range(4) if k!=key])
                keys=ks[rng.integers(0,3,size=content_length)]
            bits=rng.integers(0,2,size=content_length)
            xx[i,:content_length]=2*keys+bits
            xx[i,-1]=8+key
        elif task=='last_value':
            keys=rng.integers(0,4,size=content_length)
            bits=rng.integers(0,2,size=content_length)
            bits[-1]=y
            xx[i,:content_length]=2*keys+bits
            xx[i,-1]=12
        elif task=='last_matching_key':
            # All non-target content has non-matching keys. At least two conflicting matched values.
            ks=np.array([k for k in range(4) if k!=key])
            keys=ks[rng.integers(0,3,size=content_length)]
            bits=rng.integers(0,2,size=content_length)
            idx=np.sort(rng.choice(content_length,size=2,replace=False))
            keys[idx]=key;bits[idx[0]]=1-y;bits[idx[1]]=y
            xx[i,:content_length]=2*keys+bits
            xx[i,-1]=8+key
    return torch.from_numpy(xx),torch.from_numpy(yy)

class SharedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1=nn.LayerNorm(D)
        self.qkv=nn.Linear(D,3*D,bias=False)
        self.proj=nn.Linear(D,D)
        self.ln2=nn.LayerNorm(D)
        self.ff=nn.Sequential(nn.Linear(D,P['ff_width']),nn.SiLU(),nn.Linear(P['ff_width'],D))
    def forward(self,h,pos,arm,code,step,beta):
        b,l,d=h.shape
        q,k,v=self.qkv(self.ln1(h)).chunk(3,dim=-1)
        q=q.view(b,l,2,H).transpose(1,2)
        k=k.view(b,l,2,H).transpose(1,2)
        v=v.view(b,l,2,H).transpose(1,2)
        # We preserve the same head0 semantics across all arms; head1 semantic-value activation is the intervention.
        v0=F.silu(v[:,0])
        vv=v[:,1]
        if arm in ('mirror','fixed_mirror'):
            vv=reflect(F.silu(reflect(vv,code[step])),code[step])
        elif arm=='coupled_rank1':
            vv=rank1_trans(F.silu(rank1_trans(vv,code[step],beta[step])),code[step],beta[step])
        elif arm=='film':
            vv=F.silu(vv*(1+0.8*torch.tanh(code[step])))
        elif arm=='shared':
            vv=F.silu(vv)
        else:raise ValueError(arm)
        v=torch.stack((v0,vv),dim=1)
        if pos in ('rope','hybrid'):
            qr,kr=rotary(q[:,0]),rotary(k[:,0])
            if pos=='rope':
                q=torch.stack((qr,rotary(q[:,1])),dim=1)
                k=torch.stack((kr,rotary(k[:,1])),dim=1)
            else:
                q=torch.stack((qr,q[:,1]),dim=1)
                k=torch.stack((kr,k[:,1]),dim=1)
        scores=q@k.transpose(-2,-1)/(H**0.5)
        mask=torch.ones(l,l,device=h.device,dtype=torch.bool).triu(1)
        p=torch.softmax(scores.masked_fill(mask,float('-inf')),dim=-1)
        a=(p@v).transpose(1,2).contiguous().view(b,l,D)
        h=h+self.proj(a)
        return h+self.ff(self.ln2(h))

class Model(nn.Module):
    def __init__(self,arm,pos,seed):
        super().__init__()
        assert arm in ARMS and pos in POSITIONS
        self.arm=arm;self.pos=pos;self.seed=seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.emb=nn.Embedding(VOCAB,D)
            self.block=SharedBlock()
            self.norm=nn.LayerNorm(D)
            self.head=nn.Linear(D,2)
        if arm in ('mirror','coupled_rank1','fixed_mirror'):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(100000+seed)
                v=torch.randn(2,H)
            if arm=='fixed_mirror': self.register_buffer('code',v)
            else:self.code=nn.Parameter(v)
        elif arm=='film': self.code=nn.Parameter(torch.zeros(2,H))
        if arm=='coupled_rank1':self.beta=nn.Parameter(torch.tensor([-2.0,-2.0]))
    def forward(self,x):
        h=self.emb(x)
        for step in range(2):
            h=self.block(h,self.pos,self.arm,self.code if self.arm!='shared' else None,step,self.beta if self.arm=='coupled_rank1' else None)
        return self.head(self.norm(h[:,-1]))
    def view_difference(self,x):
        if self.arm not in ('mirror','fixed_mirror','coupled_rank1'): return None
        with torch.no_grad():
            h=self.emb(x)
            v=self.block.qkv(self.block.ln1(h)).chunk(3,dim=-1)[2]
            v=v.view(x.shape[0],x.shape[1],2,H)[:,:,1,:]
            c=self.code
            if self.arm=='coupled_rank1':
                g0=rank1_trans(F.silu(rank1_trans(v,c[0],self.beta[0])),c[0],self.beta[0])
                g1=rank1_trans(F.silu(rank1_trans(v,c[1],self.beta[1])),c[1],self.beta[1])
            else:
                g0=reflect(F.silu(reflect(v,c[0])),c[0]);g1=reflect(F.silu(reflect(v,c[1])),c[1])
            return {'v_rms':v.square().mean().sqrt().item(),
                    'view_rel_rms':((g0-g1).square().mean().sqrt()/(g0.square().mean().sqrt()+1e-12)).item(),
                    'view_cosine':F.cosine_similarity(g0.flatten(),g1.flatten(),dim=0).item(),
                    'gate_rel_rms':((g0-F.silu(v)).square().mean().sqrt()/(F.silu(v).square().mean().sqrt()+1e-12)).item()}

def bytes_inference(m):
    o=io.BytesIO();torch.save({'arch':'MA-1182','pos':m.pos,'arm':m.arm,'state':m.state_dict()},o)
    return len(o.getvalue())

def count_params(m):return sum(p.numel() for p in m.parameters())
