from __future__ import annotations
import copy, json, math, struct
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def make_world(seed:int, states:int=32, rules:int=12)->dict:
    rng=np.random.default_rng(seed)
    tables=[]
    for z in range(2):
        rows=[]
        for r in range(rules):
            rows.append(rng.permutation(states))
        tables.append(rows)
    return {'table':torch.tensor(np.asarray(tables),dtype=torch.long),'states':states,'rules':rules,'seed':seed}


def vocab_info(world:dict)->dict:
    s,r=world['states'],world['rules']
    return dict(state0=0, rule0=s, branch0=s+r, branch1=s+r+1, hidden=s+r+2, bos=s+r+3, vocab=s+r+4)


def sample_batch(world:dict,batch:int,P:int,mode:str,seed:int,forced_branch:int|None=None)->dict:
    if mode not in ('deterministic','revealed','hidden'): raise ValueError(mode)
    g=torch.Generator().manual_seed(seed)
    rules=torch.randint(world['rules'],(batch,),generator=g)
    x=torch.randint(world['states'],(batch,),generator=g)
    if forced_branch is not None:
        z=torch.full((batch,),int(forced_branch),dtype=torch.long)
    elif mode=='deterministic':
        z=torch.zeros(batch,dtype=torch.long)
    else:
        z=torch.randint(2,(batch,),generator=g)
    v=vocab_info(world)
    bz=torch.where(z==0,torch.full_like(z,v['branch0']),torch.full_like(z,v['branch1']))
    if mode=='hidden': bz=torch.full_like(z,v['hidden'])
    context=torch.stack((torch.full_like(x,v['bos']), rules+v['rule0'], bz, x),1)
    ys=[];cur=x.clone();table=world['table']
    for _ in range(P):
        cur=table[z,rules,cur]
        ys.append(cur.clone())
    return {'context':context,'targets':torch.stack(ys,1),'rules':rules,'branch':z}


def causal_mask(T:int,device)->torch.Tensor:
    m=torch.full((T,T),float('-inf'),device=device)
    return torch.triu(m,diagonal=1)


class Block(nn.Module):
    def __init__(self,d:int,heads:int,ff:int):
        super().__init__(); assert d%heads==0
        self.d=d;self.heads=heads;self.dh=d//heads
        self.n1=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d);self.out=nn.Linear(d,d)
        self.n2=nn.LayerNorm(d);self.up=nn.Linear(d,ff);self.down=nn.Linear(ff,d)
    def _split(self,z):
        b,t,d=z.shape
        return z.reshape(b,t,self.heads,self.dh).transpose(1,2)
    def full(self,x,mask):
        q,k,v=self.qkv(self.n1(x)).chunk(3,-1)
        q,k,v=map(self._split,(q,k,v))
        a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0)
        x=x+self.out(a.transpose(1,2).reshape(x.shape))
        z=self.n2(x);x=x+self.down(F.gelu(self.up(z),approximate='tanh'))
        return x
    def prefill(self,x,mask):
        q,k,v=self.qkv(self.n1(x)).chunk(3,-1)
        q,k,v=map(self._split,(q,k,v))
        a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0)
        x=x+self.out(a.transpose(1,2).reshape(x.shape))
        z=self.n2(x);x=x+self.down(F.gelu(self.up(z),approximate='tanh'))
        return x,(k,v)
    def step(self,x,cache):
        q,k,v=self.qkv(self.n1(x)).chunk(3,-1)
        q,k,v=map(self._split,(q,k,v));oldk,oldv=cache
        kk=torch.cat((oldk,k),2);vv=torch.cat((oldv,v),2)
        a=F.scaled_dot_product_attention(q,kk,vv,dropout_p=0.0)
        x=x+self.out(a.transpose(1,2).reshape(x.shape))
        z=self.n2(x);x=x+self.down(F.gelu(self.up(z),approximate='tanh'))
        return x,(kk,vv)


class TinyAR(nn.Module):
    kind='ar'
    def __init__(self,c:dict):
        super().__init__();self.config=copy.deepcopy(c);d=c['d'];self.vocab=c.get('vocab',c['states']+c['rules']+4)
        self.emb=nn.Embedding(self.vocab,d);self.pos=nn.Parameter(torch.randn(c['max_len'],d)*.02)
        self.blocks=nn.ModuleList([Block(d,c['heads'],c['ff']) for _ in range(c['layers'])])
        self.norm=nn.LayerNorm(d);self.head=nn.Linear(d,self.vocab)
    def forward(self,tokens):
        t=tokens.shape[1];x=self.emb(tokens)+self.pos[:t]
        mask=causal_mask(t,tokens.device)
        for b in self.blocks:x=b.full(x,mask)
        return self.head(self.norm(x))
    def prefill(self,tokens):
        t=tokens.shape[1];x=self.emb(tokens)+self.pos[:t];mask=causal_mask(t,tokens.device);caches=[]
        for b in self.blocks:
            x,cache=b.prefill(x,mask);caches.append(cache)
        return self.head(self.norm(x[:,-1:])),caches
    def step(self,token,caches,pos_idx:int):
        x=self.emb(token[:,None])+self.pos[pos_idx:pos_idx+1];new=[]
        for b,cache in zip(self.blocks,caches):
            x,ca=b.step(x,cache);new.append(ca)
        return self.head(self.norm(x)),new


class PeriodModel(nn.Module):
    kind='period'
    def __init__(self,c:dict,mixer:bool=True):
        super().__init__();self.config=copy.deepcopy(c);self.config['mixer']=bool(mixer);self.mixer=bool(mixer)
        d=c['d'];self.P=c['P'];self.vocab=c.get('vocab',c['states']+c['rules']+4)
        self.emb=nn.Embedding(self.vocab,d);self.pos=nn.Parameter(torch.randn(c['max_len'],d)*.02)
        self.slots=nn.Parameter(torch.randn(self.P,d)*.02)
        self.blocks=nn.ModuleList([Block(d,c['heads'],c['ff']) for _ in range(c['layers'])])
        self.norm=nn.LayerNorm(d);self.head=nn.Linear(d,self.vocab)
    def build_mask(self,C:int,P:int,device)->torch.Tensor:
        T=C+P;m=torch.full((T,T),float('-inf'),device=device)
        for i in range(C):m[i,:i+1]=0
        for i in range(P):
            q=C+i;m[q,:C]=0;m[q,q]=0
            if self.mixer and i>0:m[q,C:q]=0
        return m
    def forward(self,context):
        b,C=context.shape;P=self.P
        xctx=self.emb(context)+self.pos[:C]
        xs=self.slots[None,:,:].expand(b,-1,-1)+self.pos[C:C+P]
        x=torch.cat((xctx,xs),1);mask=self.build_mask(C,P,context.device)
        for block in self.blocks:x=block.full(x,mask)
        return self.head(self.norm(x[:,C:]))


def full_ar_logits(m:TinyAR,seq:torch.Tensor,C:int,P:int):
    logits=m(seq)
    return torch.stack([logits[:,C-1+i] for i in range(P)],1)


def cached_ar_logits(m:TinyAR,seq:torch.Tensor,C:int,P:int):
    first,caches=m.prefill(seq[:,:C]);outs=[first[:,0]]
    for i in range(1,P):
        log,caches=m.step(seq[:,C+i-1],caches,C+i-1);outs.append(log[:,0])
    return torch.stack(outs,1)


def encode_model(m:nn.Module)->bytes:
    spec=[];parts=[];offset=0
    for name,value in sorted(m.state_dict().items()):
        a=value.detach().cpu().contiguous().numpy();raw=a.tobytes();parts.append(raw)
        spec.append(dict(name=name,shape=list(a.shape),dtype=a.dtype.str,offset=offset,nbytes=len(raw)));offset+=len(raw)
    meta=dict(kind=m.kind,config=m.config,tensors=spec)
    h=json.dumps(meta,sort_keys=True,separators=(',',':')).encode()
    return b'TM001001'+struct.pack('<Q',len(h))+h+b''.join(parts)


def decode_model(raw:bytes):
    if raw[:8]!=b'TM001001':raise ValueError('bad magic')
    n=struct.unpack('<Q',raw[8:16])[0];meta=json.loads(raw[16:16+n]);body=raw[16+n:]
    with torch.random.fork_rng(devices=[]):
        m=(TinyAR(meta['config']) if meta['kind']=='ar' else (LatentPeriodModel(meta['config'],mixer=meta['config']['mixer']) if meta['kind']=='latent_period' else PeriodModel(meta['config'],mixer=meta['config']['mixer'])))
    state={};end=0
    for it in meta['tensors']:
        s=it['offset'];end=s+it['nbytes'];a=np.frombuffer(body[s:end],dtype=np.dtype(it['dtype'])).copy().reshape(it['shape']);state[it['name']]=torch.from_numpy(a)
    if end!=len(body):raise ValueError('payload length mismatch')
    m.load_state_dict(state);return m


class LatentPeriodModel(nn.Module):
    kind='latent_period'
    def __init__(self,c:dict,mixer:bool=True):
        super().__init__();self.config=copy.deepcopy(c);self.config['mixer']=bool(mixer);self.mixer=bool(mixer)
        d=c['d'];self.P=c['P'];self.K=c.get('latents',2);self.vocab=c.get('vocab',c['states']+c['rules']+4)
        self.emb=nn.Embedding(self.vocab,d);self.pos=nn.Parameter(torch.randn(c['max_len'],d)*.02)
        self.slots=nn.Parameter(torch.randn(self.P,d)*.02);self.latent=nn.Parameter(torch.randn(self.K,d)*.02)
        self.mix_logits=nn.Parameter(torch.zeros(self.K))
        self.blocks=nn.ModuleList([Block(d,c['heads'],c['ff']) for _ in range(c['layers'])])
        self.norm=nn.LayerNorm(d);self.head=nn.Linear(d,self.vocab)
    def build_mask(self,C:int,P:int,device):
        T=C+P;m=torch.full((T,T),float('-inf'),device=device)
        for i in range(C):m[i,:i+1]=0
        for i in range(P):
            q=C+i;m[q,:C]=0;m[q,q]=0
            if self.mixer and i>0:m[q,C:q]=0
        return m
    def _run(self,context,latent_vec):
        b,C=context.shape;P=self.P
        xctx=self.emb(context)+self.pos[:C]
        xs=self.slots[None,:,:]+latent_vec[:,None,:]+self.pos[C:C+P]
        x=torch.cat((xctx,xs),1);mask=self.build_mask(C,P,context.device)
        for block in self.blocks:x=block.full(x,mask)
        return self.head(self.norm(x[:,C:]))
    def forward(self,context,latent_id:torch.Tensor|None=None):
        b=context.shape[0]
        if latent_id is not None:
            return self._run(context,self.latent[latent_id])
        cx=context[:,None,:].expand(b,self.K,-1).reshape(b*self.K,-1)
        lv=self.latent[None,:,:].expand(b,-1,-1).reshape(b*self.K,-1)
        y=self._run(cx,lv)
        return y.reshape(b,self.K,self.P,self.vocab)
    def mixture_logits(self,batch:int):
        return self.mix_logits[None,:].expand(batch,-1)


def latent_joint_nll(logits:torch.Tensor,target:torch.Tensor,mix_logits:torch.Tensor,reduction:str='mean'):
    logp=F.log_softmax(logits,-1)
    idx=target[:,None,:,None].expand(-1,logits.shape[1],-1,1)
    seq_lp=logp.gather(-1,idx).squeeze(-1).sum(-1)
    joint=torch.logsumexp(F.log_softmax(mix_logits,-1)+seq_lp,dim=-1)
    nll=-joint
    if reduction=='none':return nll
    if reduction=='sum':return nll.sum()
    return nll.mean()


def latent_hard_em_loss(logits:torch.Tensor,target:torch.Tensor,balance_weight:float=0.05):
    logp=F.log_softmax(logits,-1)
    idx=target[:,None,:,None].expand(-1,logits.shape[1],-1,1)
    seq_nll=-logp.gather(-1,idx).squeeze(-1).sum(-1)
    best,assign=seq_nll.min(-1)
    loss=best.mean()
    if balance_weight>0 and logits.shape[1]>1:
        frac=torch.stack([(assign==k).float().mean() for k in range(logits.shape[1])])
        loss=loss+balance_weight*((frac-1.0/logits.shape[1])**2).sum()
    return loss,assign
