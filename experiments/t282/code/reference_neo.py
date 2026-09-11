"""Independent nn.Module GPT-Neo reference, informed by HF v4.28.1.
NOT an installed Transformers runtime. Uses checkpoint-stored attention masks,
nn.Module projections, native tanh-GELU and optional K/V cache. No engine import.
Source: huggingface/transformers/src/transformers/models/gpt_neo/modeling_gpt_neo.py
v4.28.1, file Git blob 70bfb7e45a7c3196f258b0babbd581bc4fc42bbe.
Only batch1, no padding, tied embeddings, FP32 inference is supported.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

def linear(state,p):
    w=state[p+'.weight'];m=nn.Linear(w.shape[1],w.shape[0],bias=p+'.bias' in state,device='meta')
    m.weight=nn.Parameter(w,requires_grad=False)
    if p+'.bias' in state:m.bias=nn.Parameter(state[p+'.bias'],requires_grad=False)
    return m

def layernorm(state,p,eps):
    m=nn.LayerNorm(state[p+'.weight'].shape,eps=eps,device='meta')
    m.weight=nn.Parameter(state[p+'.weight'],requires_grad=False);m.bias=nn.Parameter(state[p+'.bias'],requires_grad=False)
    return m

class ReferenceBlock(nn.Module):
    def __init__(self,state,raw,cfg,i):
        super().__init__();p=f'transformer.h.{i}';a=p+'.attn.attention'
        self.nh=cfg['num_heads'];self.dh=cfg['hidden_size']//self.nh;self.local=cfg['attention_layers'][i]=='local'
        self.ln1=layernorm(state,p+'.ln_1',cfg['layer_norm_epsilon']);self.ln2=layernorm(state,p+'.ln_2',cfg['layer_norm_epsilon'])
        self.q=linear(state,a+'.q_proj');self.k=linear(state,a+'.k_proj');self.v=linear(state,a+'.v_proj');self.out=linear(state,a+'.out_proj')
        self.fc=linear(state,p+'.mlp.c_fc');self.proj=linear(state,p+'.mlp.c_proj')
        self.register_buffer('mask',raw[a+'.bias'].to(torch.bool))
    def forward(self,x,offset,past=None,wrong_scale=False,wrong_global=False):
        a=self.ln1(x);b,t,_=a.shape
        q=self.q(a).reshape(b,t,self.nh,self.dh).permute(0,2,1,3)
        k=self.k(a).reshape(b,t,self.nh,self.dh).permute(0,2,1,3)
        v=self.v(a).reshape(b,t,self.nh,self.dh).permute(0,2,1,3)
        if past is not None:k=torch.cat((past[0],k),-2);v=torch.cat((past[1],v),-2)
        total=k.shape[-2];scores=torch.matmul(q.float(),k.float().transpose(-1,-2))
        if wrong_scale:scores=scores/math.sqrt(self.dh)
        if wrong_global:
            posq=torch.arange(offset,offset+t,device=x.device)[:,None];posk=torch.arange(total,device=x.device)[None,:]
            allowed=(posk<=posq)[None,None]
        else:allowed=self.mask[...,offset:offset+t,:total]
        scores=torch.where(allowed,scores,torch.tensor(torch.finfo(scores.dtype).min,device=x.device))
        probs=F.softmax(scores,-1).to(v.dtype)
        y=torch.matmul(probs,v).permute(0,2,1,3).contiguous().reshape(b,t,-1)
        x=self.out(y)+x
        return x+self.proj(F.gelu(self.fc(self.ln2(x)),approximate='tanh')),(k,v)

class ReferenceNeo(nn.Module):
    def __init__(self,state,raw,cfg):
        super().__init__();self.cfg=cfg
        self.wte=nn.Embedding.from_pretrained(state['transformer.wte.weight'],freeze=True)
        self.wpe=nn.Embedding.from_pretrained(state['transformer.wpe.weight'],freeze=True)
        self.layers=nn.ModuleList([ReferenceBlock(state,raw,cfg,i) for i in range(cfg['num_layers'])])
        self.final=layernorm(state,'transformer.ln_f',cfg['layer_norm_epsilon']);self.eval()
    def forward(self,ids,past=None,wrong_scale=False,wrong_global=False):
        if ids.ndim!=2 or ids.shape[0]!=1 or not ids.shape[1]:raise ValueError('expected nonempty batch1')
        offset=0 if past is None else past[0][0].shape[-2]
        if offset+ids.shape[1]>self.cfg['max_position_embeddings']:raise ValueError('position overflow')
        positions=torch.arange(offset,offset+ids.shape[1],device=ids.device)[None]
        x=self.wte(ids)+self.wpe(positions);cache=[]
        for i,layer in enumerate(self.layers):
            x,p=layer(x,offset,None if past is None else past[i],wrong_scale,wrong_global);cache.append(p)
        return F.linear(self.final(x),self.wte.weight),cache
