"""SRM003: causal endpoint-token learning; no oracle IDs/masks enter Model.forward.

All neural quantities are dimensionless FP32. Data tables are research data,
not inference payload. PyTorch CPU eager is the reference implementation.
"""
from __future__ import annotations
import copy, hashlib, itertools, json, math, struct
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def make_world(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    bits = ((np.arange(16)[:, None] >> np.arange(4)) & 1)
    affine = []
    for perm in itertools.permutations(range(4)):
        for mask in range(16):
            affine.append(tuple(((bits[:, perm] * (1 << np.arange(4))).sum(1) ^ mask).tolist()))
    available = list(dict.fromkeys(affine))
    chosen = rng.choice(len(available), 18, replace=False)
    rules = [available[i] for i in chosen]
    excluded = set(available)
    while len(rules) < 24:
        candidate = tuple(rng.permutation(16).tolist())
        if candidate not in excluded:
            rules.append(candidate); excluded.add(candidate)
    order = rng.permutation(24)
    table = torch.tensor(np.array(rules)[order], dtype=torch.long)
    private = torch.tensor(order >= 18)
    result = {'table': table, 'private_truth': private, 'world_seed': seed}
    for split in ('train', 'dev', 'audit'):
        pairs = []
        for a in range(24):
            for b in range(24):
                if a == b: continue
                key = f'{seed}:{min(a,b)}:{max(a,b)}'.encode()
                value = int.from_bytes(hashlib.sha256(key).digest()[:4], 'little') % 10
                group = 'audit' if value < 2 else ('dev' if value == 2 else 'train')
                if group == split: pairs.append((a,b))
        pair = torch.tensor(pairs, dtype=torch.long)
        a,b = pair.repeat_interleave(16, 0).T
        x = torch.arange(16).repeat(len(pair))
        result[split+'_pairs'] = pair
        result[split+'_tokens'] = torch.stack((torch.ones_like(x),x+3,a+19,b+19,torch.full_like(x,2)),1)
        result[split+'_targets'] = 3+table[b,table[a,x]]
    r = torch.arange(24).repeat_interleave(16); x = torch.arange(16).repeat(24)
    result['atomic_tokens'] = torch.stack((torch.ones_like(x),x+3,r+19,torch.full_like(x,2),torch.zeros_like(x)),1)
    result['atomic_targets'] = 3+table[r,x]
    return result


class FFN(nn.Module):
    def __init__(self, d: int, width: int):
        super().__init__(); self.up=nn.Linear(d,width); self.down=nn.Linear(width,d)
    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate='tanh'))


class SparseBank(nn.Module):
    """Actually gather and compute only top-k experts (router logits remain dense)."""
    def __init__(self,d:int,n:int,rank:int,k:int,signed:bool=False,nonlinear:bool=False):
        super().__init__(); self.n=n; self.k=k; self.signed=signed; self.nonlinear=nonlinear
        self.router=nn.Linear(d,max(n,1))
        if n == 0:
            self.router.weight=nn.Parameter(torch.empty(0,d));self.router.bias=nn.Parameter(torch.empty(0));self.router.out_features=0
        self.down=nn.Parameter(torch.randn(n,d,rank)/math.sqrt(d))
        self.up=nn.Parameter(torch.zeros(n,rank,d))
        self.register_buffer('slot_ids',torch.arange(n))
        if nonlinear:
            self.bias1=nn.Parameter(torch.zeros(n,rank));self.bias2=nn.Parameter(torch.zeros(n,d))
    def forward(self,x):
        if self.n == 0: return torch.zeros_like(x)
        shape=x.shape;flat=x.reshape(-1,shape[-1]);scores=self.router(flat)
        idx=(scores.abs() if self.signed else scores).topk(min(self.k,self.n),dim=-1).indices
        chosen=scores.gather(1,idx)
        if self.signed:
            coeff=torch.tanh(chosen)/min(self.k,self.n)
        else:
            coeff=chosen.softmax(-1)
        h=torch.einsum('bd,bkdr->bkr',flat,self.down[idx])
        if self.nonlinear: h=F.gelu(h+self.bias1[idx],approximate='tanh')
        y=torch.einsum('bkr,bkrd->bkd',h,self.up[idx])
        if self.nonlinear: y=y+self.bias2[idx]
        return (y*coeff[...,None]).sum(1).reshape(shape)


class Residual(nn.Module):
    def __init__(self,c:dict):
        super().__init__();d=c['d'];kind=c['kind']
        self.shared=None;self.private=None
        if kind in ('shared','hybrid'):
            self.shared=SparseBank(d,c['atoms'],c['arank'],c['topk'],signed=True)
        if kind in ('moe','lora','hybrid'):
            rank=c['expert_ff'] if kind=='moe' else c['rank']
            self.private=SparseBank(d,c['experts'],rank,c['topk'],nonlinear=kind=='moe')
    def forward(self,x):
        y=torch.zeros_like(x)
        if self.shared is not None:y=y+self.shared(x)
        if self.private is not None:y=y+self.private(x)
        return y


class Block(nn.Module):
    def __init__(self,c:dict,width:int):
        super().__init__();d=c['d'];self.heads=c['heads']
        self.norm1=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d);self.out=nn.Linear(d,d)
        self.norm2=nn.LayerNorm(d);self.ff=FFN(d,width)
    def attention(self,x):
        b,t,d=x.shape
        q,k,v=self.qkv(self.norm1(x)).chunk(3,-1)
        q,k,v=[z.reshape(b,t,self.heads,d//self.heads).transpose(1,2) for z in (q,k,v)]
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.0)
        return x+self.out(a.transpose(1,2).reshape(b,t,d))


class Model(nn.Module):
    def __init__(self,c:dict):
        super().__init__();self.config=copy.deepcopy(c)
        self.emb=nn.Embedding(c['vocab'],c['d']);self.pos=nn.Parameter(torch.randn(c['length'],c['d'])*.02)
        self.blocks=nn.ModuleList([Block(c,c['final_ff'] if i==c['layers']-1 else c['ff']) for i in range(c['layers'])])
        self.residual=Residual(c);self.norm=nn.LayerNorm(c['d']);self.head=nn.Linear(c['d'],c['vocab'])
    def forward(self,tokens:torch.Tensor,return_all:bool=False):
        x=self.emb(tokens)+self.pos[:tokens.shape[1]]
        for i,block in enumerate(self.blocks):
            x=block.attention(x);z=block.norm2(x)
            x=x+block.ff(z)
            if i==len(self.blocks)-1:x=x+self.residual(z)
        logits=self.head(self.norm(x))
        if return_all:return logits
        query=(tokens==2).to(torch.long).argmax(1)
        return logits[torch.arange(len(tokens),device=tokens.device),query]


def make_model(c:dict,seed:int,parent:Model|None=None)->Model:
    torch.manual_seed(seed);m=Model(c)
    if parent is not None:
        source=parent.state_dict();dest=m.state_dict()
        for name,value in source.items():
            if name in dest and dest[name].shape==value.shape:dest[name].copy_(value)
        # Net2Wider-style zero-output insertion for a larger dense final FFN.
        old=parent.blocks[-1].ff;new=m.blocks[-1].ff;n=old.up.out_features
        if new.up.out_features>n:
            with torch.no_grad():
                new.up.weight[:n].copy_(old.up.weight);new.up.bias[:n].copy_(old.up.bias)
                new.down.weight.zero_();new.down.weight[:,:n].copy_(old.down.weight)
                new.down.bias.copy_(old.down.bias)
    return m


def encode_model(m:Model)->bytes:
    parts=[];spec=[];offset=0
    for name,value in sorted(m.state_dict().items()):
        a=value.detach().cpu().contiguous().numpy()
        a=a.astype(a.dtype.newbyteorder('<'),copy=False)
        raw=a.tobytes();parts.append(raw)
        spec.append(dict(name=name,shape=list(a.shape),dtype=a.dtype.str,offset=offset,nbytes=len(raw)))
        offset+=len(raw)
    header=json.dumps(dict(config=m.config,tensors=spec),sort_keys=True,separators=(',',':')).encode()
    return b'SRM00301'+struct.pack('<Q',len(header))+header+b''.join(parts)


def decode_model(raw:bytes)->Model:
    if raw[:8]!=b'SRM00301':raise ValueError('bad magic')
    hlen=struct.unpack('<Q',raw[8:16])[0];meta=json.loads(raw[16:16+hlen]);body=raw[16+hlen:]
    # Constructor random initialization must not alter training RNG during audits.
    with torch.random.fork_rng(devices=[]):m=Model(meta['config'])
    state={};end=0
    for item in meta['tensors']:
        start=item['offset'];end=start+item['nbytes']
        if end>len(body):raise ValueError('truncated payload')
        a=np.frombuffer(body[start:end],dtype=item['dtype']).copy().reshape(item['shape'])
        state[item['name']]=torch.from_numpy(a)
    if end!=len(body):raise ValueError('extra payload')
    m.load_state_dict(state);return m


def compact(m:Model,keep:list[int])->Model:
    old=m.residual.private
    if old is None:raise ValueError('no private bank')
    if len(set(keep))!=len(keep) or any(i<0 or i>=old.n for i in keep):raise ValueError('invalid support')
    c=copy.deepcopy(m.config);c['experts']=len(keep)
    with torch.random.fork_rng(devices=[]):n=Model(c)
    state=m.state_dict();selected={}
    for name,value in state.items():
        selected[name]=value[keep].clone() if name.startswith('residual.private.') else value.clone()
    n.load_state_dict(selected);return n
