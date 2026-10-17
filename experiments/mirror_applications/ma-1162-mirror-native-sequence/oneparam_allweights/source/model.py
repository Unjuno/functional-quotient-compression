"""MA-1162 one-scalar-address all-weights causal Transformer feasibility model."""
import io
import math
from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch.nn import functional as F

MODES = ('mirror', 'shared', 'gain', 'dense', 'dense_byte', 'scalar_only')


@dataclass(frozen=True)
class Config:
    vocab: int = 64
    length: int = 16
    d: int = 32
    ff: int = 64
    layers: int = 2
    heads: int = 2
    root: int = 64


def rotate_right(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Apply a one-parameter orthogonal group to every row; no learned generator."""
    n = x.shape[-1]
    if n % 2:
        raise ValueError('rotation dimension must be even')
    freq = torch.arange(1, n//2+1, dtype=x.dtype, device=x.device)
    phase = angle[..., None, None] * freq
    c, s = phase.cos(), phase.sin()
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a*c-b*s, a*s+b*c), dim=-1).flatten(-2)


def conjugate(core: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Batch of R(angle)^T core R(angle), one scalar per returned full matrix."""
    right = rotate_right(core[None], angles)
    return rotate_right(right.transpose(-1, -2), angles).transpose(-1, -2)


class Transformer(nn.Module):
    """No independent embedding, FFN, readout, bias or norm weight in Mirror mode.

    `scalar_only` has one trainable SCALAR, not merely one Parameter object.
    The frozen random core in that control is serialized, not treated as free.
    """
    def __init__(self, mode: str = 'mirror', config: Config | None = None):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.cfg = config or (Config(d=12, ff=30) if mode == 'dense_byte' else Config())
        c = self.cfg
        if c.d % c.heads or c.d % 2 or max(c.vocab,c.d,c.ff)>c.root:
            raise ValueError('invalid feature/head/root dimensions')
        self.shapes = [(c.vocab,c.d)]
        for _ in range(c.layers):
            self.shapes += [(c.d,c.d)]*4 + [(c.d,c.ff),(c.ff,c.d)]
        self.shapes += [(c.d,c.vocab)]
        self.roles = len(self.shapes)
        self.register_buffer('offsets', torch.arange(self.roles)*2.399963229728653, persistent=False)
        pos = torch.arange(c.length,dtype=torch.float32)[:,None]
        rate = torch.exp(-math.log(10000)*torch.arange(0,c.d,2)/c.d)
        pe = torch.stack(((pos*rate).sin(),(pos*rate).cos()),dim=-1).flatten(-2)
        self.register_buffer('positions',pe,persistent=False)
        self.register_buffer('mask',torch.ones(c.length,c.length,dtype=torch.bool).triu(1),persistent=False)
        if mode.startswith('dense'):
            self.weights = nn.ParameterList([
                nn.Parameter(torch.randn(a,b)*(.5 if i==0 else a**-.5))
                for i,(a,b) in enumerate(self.shapes)])
        else:
            root = torch.randn(c.root,c.root)/math.sqrt(c.root)
            if mode == 'scalar_only':
                self.register_buffer('core',root)
                self.codes = nn.Parameter(torch.zeros(1))
            else:
                self.core = nn.Parameter(root)
                if mode in ('mirror','gain'):
                    self.codes = nn.Parameter(self.offsets.clone() if mode=='mirror' else torch.zeros(self.roles))

    def matrices(self) -> list[torch.Tensor]:
        if self.mode.startswith('dense'):
            return list(self.weights)
        if self.mode in ('mirror','scalar_only'):
            angles = self.codes if self.mode=='mirror' else self.offsets + self.codes[0]
            full = conjugate(self.core, angles)
        else:
            full = self.core[None].expand(self.roles,-1,-1)
        out = []
        for i,(a,b) in enumerate(self.shapes):
            scale = .5*math.sqrt(self.cfg.root) if i==0 else math.sqrt(self.cfg.root/a)
            w = full[i,:a,:b] * scale
            if self.mode=='gain':
                w = w * self.codes[i].exp()
            out.append(w)
        return out

    def forward(self, tokens: torch.Tensor, all_positions: bool = False) -> torch.Tensor:
        if tokens.ndim!=2 or tokens.dtype!=torch.long or tokens.shape[1]>self.cfg.length:
            raise ValueError('expected int64 [batch,length<=configured length]')
        c=self.cfg; b,l=tokens.shape
        w=self.matrices()
        h=F.embedding(tokens,w[0])+self.positions[:l]
        for layer in range(c.layers):
            i=1+layer*6
            z=F.layer_norm(h,(c.d,),eps=1e-5)
            q,k,v=[(z @ w[i+j]).reshape(b,l,c.heads,c.d//c.heads).transpose(1,2) for j in range(3)]
            att=(q @ k.transpose(-1,-2))/(c.d//c.heads)**.5
            att=att.masked_fill(self.mask[:l,:l],-torch.inf).softmax(-1)
            mixed=(att @ v).transpose(1,2).reshape(b,l,c.d)
            h=h+mixed @ w[i+3]
            z=F.layer_norm(h,(c.d,),eps=1e-5)
            h=h+F.gelu(z @ w[i+4]) @ w[i+5]
        h=F.layer_norm(h,(c.d,),eps=1e-5)
        return (h if all_positions else h[:,-1]) @ w[-1]


def pack(model: Transformer) -> bytes:
    """Actual compact inference serialization; includes frozen core in scalar-only.

    Use one flattened state tensor for ALL methods to avoid per-tensor ZIP padding
    becoming an artificial compression advantage. Config implies layout/order.
    No optimizer, no stored temporary/generated matrices, no uncharged fixed core.
    """
    state=model.state_dict()
    flat=torch.cat([v.detach().cpu().reshape(-1) for v in state.values()])
    payload={'schema':1,'mode':model.mode,'config':asdict(model.cfg),'flat':flat}
    buf=io.BytesIO(); torch.save(payload,buf)
    return buf.getvalue()


def unpack(data: bytes) -> Transformer:
    obj=torch.load(io.BytesIO(data),weights_only=True,map_location='cpu')
    if obj['schema']!=1:
        raise ValueError('unsupported schema')
    with torch.random.fork_rng():
        model=Transformer(obj['mode'],Config(**obj['config']))
    state=model.state_dict(); offset=0
    for key,target in state.items():
        size=target.numel()
        state[key]=obj['flat'][offset:offset+size].reshape_as(target).clone()
        offset+=size
    if offset!=obj['flat'].numel():
        raise ValueError('unexpected payload size')
    model.load_state_dict(state)
    return model.eval()


def world_encoding(world: int) -> torch.Tensor:
    return torch.randperm(64,generator=torch.Generator().manual_seed(world))


def batch_data(task: str, batch: int, gen: torch.Generator, perm: torch.Tensor):
    if task=='copy':
        prefix=torch.randint(0,8,(batch,15),generator=gen)
        lag=torch.randint(1,5,(batch,),generator=gen)
        target=prefix.gather(1,(15-lag)[:,None]).squeeze(1)
        raw=torch.cat((prefix,(48+lag)[:,None]),1)
    elif task=='retrieve':
        keys=torch.rand(batch,8,generator=gen).argsort(-1)
        values=torch.randint(0,4,(batch,8),generator=gen)
        index=torch.randint(0,8,(batch,),generator=gen)
        query=keys.gather(1,index[:,None]).squeeze(1)
        target=values.gather(1,index[:,None]).squeeze(1)
        raw=torch.cat((keys*4+values,torch.full((batch,7),63), (32+query)[:,None]),1)
    else:
        raise ValueError(task)
    return perm[raw],target
