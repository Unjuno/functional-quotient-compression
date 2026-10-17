"""MA-1162 ONE-CORE: all attention/FFN/embeddings/output from ONE nn.Parameter.

Mirror views are orthogonal conjugates of the single physical weight matrix S.
Even the role addresses are obtained differentiably from entries of S, never
from separate nn.Parameter/Embedding/Linear/affine LayerNorm modules.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import io
import math
import torch
from torch import nn
from torch.nn import functional as F

MODES = ('mirror', 'mirror_static', 'shared', 'gain', 'index', 'local', 'dense_match', 'dense_full')


@dataclass(frozen=True)
class Config:
    vocab: int = 64
    length: int = 16
    d: int = 32
    ff: int = 64
    layers: int = 2
    heads: int = 2
    root: int = 64


def rotate(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Right-multiply rows by real orthogonal block rotations with known frequency."""
    n = x.size(-1)
    if n % 2:
        raise ValueError('rotation requires even width')
    freq = torch.arange(1, n//2 + 1, dtype=x.dtype, device=x.device)
    phase = angle[..., None, None] * freq
    c, s = phase.cos(), phase.sin()
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((a*c - b*s, a*s + b*c), dim=-1).flatten(-2)


def conjugate(s: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Matrix orbit Q(a)^T S Q(a), one full tensor per role in one batched op."""
    right = rotate(s[None], angles)
    return rotate(right.transpose(-1, -2), angles).transpose(-1, -2)


class SingleCoreTransformer(nn.Module):
    def __init__(self, mode: str = 'mirror', config: Config | None = None):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        if config is None and mode == 'dense_match':
            config = Config(d=12, ff=30)
        self.cfg, self.mode = config or Config(), mode
        c = self.cfg
        if c.d % c.heads or c.d % 2 or c.ff % 2 or max(c.vocab,c.d,c.ff) > c.root:
            raise ValueError('invalid model dimensions')
        self.shapes = [(c.vocab, c.d)]
        for _ in range(c.layers):
            self.shapes += [(c.d, c.d)]*4 + [(c.d, c.ff), (c.ff, c.d)]
        self.shapes += [(c.d,c.vocab)]
        self.roles = len(self.shapes)
        if self.roles > c.root:
            raise ValueError('root must store at least one diagonal code entry per role')
        # Fully deterministic, reconstructible metadata: no extra stored model weights.
        offsets = torch.arange(self.roles, dtype=torch.float32) * 0.17320508075688773
        self.register_buffer('offsets', offsets, persistent=False)
        t = torch.arange(c.length, dtype=torch.float32)[:, None]
        inv = torch.exp(-math.log(10000) * torch.arange(0,c.d,2) / c.d)
        pos = torch.stack(((t*inv).sin(), (t*inv).cos()), -1).flatten(-2)
        self.register_buffer('position', pos, persistent=False)
        self.register_buffer('mask', torch.ones(c.length,c.length,dtype=torch.bool).triu(1), persistent=False)
        if mode.startswith('dense'):
            self.weights = nn.ParameterList([nn.Parameter(torch.randn(a,b)/math.sqrt(a))
                                             for a,b in self.shapes])
        else:
            self.S = nn.Parameter(torch.randn(c.root,c.root) / math.sqrt(c.root))
        # No independently trainable code, embedding, QKV, FFN, readout, norms or biases.

    def angles(self) -> torch.Tensor:
        if self.mode == 'mirror':
            # m is a view of S, NOT a new trainable tensor/physical parameter.
            return self.offsets + 0.3 * torch.tanh(self.S.diagonal()[:self.roles])
        if self.mode == 'mirror_static':
            return self.offsets
        raise ValueError('only Mirror models have rotation codes')

    def virtual_matrices(self) -> list[torch.Tensor]:
        if self.mode.startswith('dense'):
            return list(self.weights)
        s=self.S
        if self.mode.startswith('mirror'):
            all_w=conjugate(s,self.angles())
        elif self.mode == 'index':
            all_w=torch.stack([torch.roll(s,(i*7,i*13),(0,1)) for i in range(self.roles)])
        else:
            all_w=s.unsqueeze(0).expand(self.roles,-1,-1)
        out=[]
        for j,(a,b) in enumerate(self.shapes):
            # Equal deterministic fan-in scaling for shared and Mirror models.
            scale=.5*math.sqrt(self.cfg.root) if j==0 else math.sqrt(self.cfg.root/a)
            m=all_w[j,:a,:b] * scale
            if self.mode=='gain':
                # Ordinary scalar per-role gain read from S, no extra parameters.
                m=m*torch.exp(.3*torch.tanh(s.diagonal()[j]))
            out.append(m)
        return out

    def forward(self, token_ids: torch.Tensor, all_positions: bool = False) -> torch.Tensor:
        if token_ids.dtype != torch.long or token_ids.ndim!=2 or token_ids.shape[1]>self.cfg.length:
            raise ValueError('expect int64 [B,L<=length]')
        c=self.cfg
        batch, l=token_ids.shape
        matrices=self.virtual_matrices()
        h=F.embedding(token_ids, matrices[0]) + self.position[:l]
        for layer in range(c.layers):
            i=1+6*layer
            if self.mode != 'local':
                z=F.layer_norm(h, (c.d,), eps=1e-5)
                q,k,v=[(z@matrices[i+j]).reshape(batch,l,c.heads,c.d//c.heads).transpose(1,2)
                       for j in range(3)]
                a=(q@k.transpose(-1,-2)) / math.sqrt(c.d//c.heads)
                a=torch.softmax(a.masked_fill(self.mask[:l,:l], -torch.inf),dim=-1)
                h=h+(a@v).transpose(1,2).reshape(batch,l,c.d)@matrices[i+3]
            z=F.layer_norm(h,(c.d,),eps=1e-5)
            h=h+F.gelu(z@matrices[i+4])@matrices[i+5]
        logits=F.layer_norm(h,(c.d,),eps=1e-5)@matrices[-1]
        return logits if all_positions else logits[:,-1,:]


def packed_bytes(model: SingleCoreTransformer) -> bytes:
    """Serialize all inference state, flattening to avoid per-tensor zip overhead advantage."""
    flat=torch.cat([p.detach().cpu().flatten() for p in model.state_dict().values()])
    blob={'schema':1, 'mode':model.mode, 'cfg':asdict(model.cfg), 'state':flat}
    stream=io.BytesIO()
    torch.save(blob, stream)
    return stream.getvalue()


def unpacked(data: bytes) -> SingleCoreTransformer:
    o=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
    if o['schema'] != 1:
        raise ValueError('unsupported schema')
    with torch.random.fork_rng():
        model=SingleCoreTransformer(o['mode'],Config(**o['cfg']))
    state=model.state_dict()
    pos=0
    for name,p in state.items():
        n=p.numel()
        state[name]=o['state'][pos:pos+n].reshape_as(p).clone()
        pos+=n
    if pos != o['state'].numel():
        raise ValueError('unexpected state count')
    model.load_state_dict(state)
    return model.eval()


def world_map(world: int):
    return torch.randperm(64,generator=torch.Generator().manual_seed(int(world)))


def make_batch(task: str, batch: int, rng: torch.Generator, world: torch.Tensor):
    """Last-position query; target is a function of previously observed tokens."""
    if task=='copy':
        prefix=torch.randint(0,8,(batch,15),generator=rng)
        lag=torch.randint(1,5,(batch,),generator=rng)
        y=prefix.gather(1,(15-lag)[:,None]).squeeze(1)
        x=torch.cat((prefix,(48+lag)[:,None]),1)
    elif task=='retrieve':
        keys=torch.rand(batch,8,generator=rng).argsort(-1)
        vals=torch.randint(0,4,(batch,8),generator=rng)
        idx=torch.randint(0,8,(batch,),generator=rng)
        query=keys.gather(1,idx[:,None]).squeeze(1)
        y=vals.gather(1,idx[:,None]).squeeze(1)
        x=torch.cat((keys*4+vals,torch.full((batch,7),63),(32+query)[:,None]),1)
    elif task=='hidden':
        prefix=torch.randint(0,8,(batch,15),generator=rng)
        x=torch.cat((prefix,torch.full((batch,1),62)),1)
        y=torch.randint(0,4,(batch,),generator=rng)
    else:
        raise ValueError(task)
    return world[x], y


def nlearned(model):
    return sum(p.numel() for p in model.parameters())


def logical_params(model):
    return sum(a*b for a,b in model.shapes)


def dense_macs_proxy(model, l=16):
    c=model.cfg
    att=0 if model.mode=='local' else 4*l*c.d*c.d+2*l*l*c.d
    return c.layers*(att+2*l*c.d*c.ff) + l*c.d*c.vocab + c.d*c.vocab