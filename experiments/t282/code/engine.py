"""Offline GPT-Neo inference and byte-level BPE for uploaded TinyStories checkpoints.
Scope: no padding, batch=1, no cache, FP32. Tied output embeddings.
Source specification: HF transformers v4.28.1 GPT-Neo; not HF runtime parity certified.
"""
from __future__ import annotations
import json, math, hashlib
from pathlib import Path
from functools import lru_cache
import regex
import torch
import torch.nn.functional as F

class BPETokenizer:
    def __init__(self, model_dir: str | Path):
        p=Path(model_dir)
        obj=json.loads((p/'tokenizer.json').read_text())
        self.vocab=obj['model']['vocab']; self.inverse={v:k for k,v in self.vocab.items()}
        assert self.vocab == json.loads((p/'vocab.json').read_text())
        self.ranks={tuple(m.split(' ')):i for i,m in enumerate(obj['model']['merges'])}
        bs=list(range(33,127))+list(range(161,173))+list(range(174,256)); cs=bs.copy(); n=0
        for b in range(256):
            if b not in bs: bs.append(b); cs.append(256+n); n+=1
        self.byte_to_char=dict(zip(bs,map(chr,cs)))
        self.char_to_byte={c:b for b,c in self.byte_to_char.items()}
        self.pattern=regex.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
    @lru_cache(maxsize=20000)
    def bpe(self, word: str) -> tuple[str,...]:
        parts=tuple(word)
        while len(parts)>1:
            pair=min(zip(parts,parts[1:]), key=lambda p:self.ranks.get(p,float('inf')))
            if pair not in self.ranks:break
            new=[];i=0
            while i<len(parts):
                if i+1<len(parts) and (parts[i],parts[i+1])==pair:
                    new.append(parts[i]+parts[i+1]);i+=2
                else:new.append(parts[i]);i+=1
            parts=tuple(new)
        return parts
    def encode(self,text:str)->list[int]:
        ids=[]
        for p in self.pattern.findall(text):
            mapped=''.join(self.byte_to_char[b] for b in p.encode('utf-8'))
            ids.extend(self.vocab[s] for s in self.bpe(mapped))
        return ids
    def decode(self,ids:list[int])->str:
        return bytes(self.char_to_byte[c] for i in ids for c in self.inverse[i]).decode('utf-8',errors='replace')

def load_checkpoint(model_dir: str | Path, device: str='cpu'):
    p=Path(model_dir); cfg=json.loads((p/'config.json').read_text())
    raw=torch.load(p/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    state={k:v.to(device=device,dtype=torch.float32) for k,v in raw.items()
           if not k.endswith('.attn.attention.bias') and not k.endswith('.masked_bias')}
    assert 'lm_head.weight' not in state, 'This path assumes tied embeddings, no separate LM head.'
    return cfg,state,raw

def gelu_new(x):
    return 0.5*x*(1.0+torch.tanh(math.sqrt(2.0/math.pi)*(x+0.044715*torch.pow(x,3.0))))

def forward(state:dict, cfg:dict, ids:torch.Tensor, backend:str='eager', collect:bool=False):
    if ids.ndim!=2 or ids.shape[0]!=1:raise ValueError('batch=1, no padded sequences')
    length=ids.shape[1]; hidden=cfg['hidden_size']; heads=cfg['num_heads']
    if length>cfg['max_position_embeddings']:raise ValueError('context overflow')
    if cfg['activation_function']!='gelu_new':raise ValueError('unsupported activation')
    pos=torch.arange(length,device=ids.device)[None,:]
    x=F.embedding(ids,state['transformer.wte.weight'])+F.embedding(pos,state['transformer.wpe.weight'])
    mask=torch.ones((length,length),dtype=torch.bool,device=ids.device).tril()
    cache={}
    def ln(a,p):return F.layer_norm(a,(hidden,),state[p+'.weight'],state[p+'.bias'],cfg['layer_norm_epsilon'])
    def linear(a,p):return F.linear(a,state[p+'.weight'],state.get(p+'.bias'))
    for i in range(cfg['num_layers']):
        p=f'transformer.h.{i}'; a=ln(x,p+'.ln_1'); att=p+'.attn.attention'
        q,k,v=[linear(a,att+'.'+r+'_proj').view(1,length,heads,hidden//heads).transpose(1,2) for r in ['q','k','v']]
        usemask=mask
        if cfg['attention_layers'][i]=='local':usemask=mask ^ mask.tril(-cfg['window_size'])
        if backend=='eager':
            # GPT-Neo uses UN-SCALED dot-product attention. Do not insert sqrt(head_dim).
            scores=q.float()@k.float().transpose(-1,-2)
            scores=scores.masked_fill(~usemask,torch.finfo(torch.float32).min)
            y=torch.softmax(scores,dim=-1).to(v.dtype)@v
        elif backend=='sdpa':
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=usemask,dropout_p=0.0,scale=1.0)
        else:raise ValueError(backend)
        y=y.transpose(1,2).contiguous().view(1,length,hidden)
        x=x+linear(y,att+'.out_proj')
        a=ln(x,p+'.ln_2'); f=linear(a,p+'.mlp.c_fc')
        act=gelu_new(f) if backend=='eager' else F.gelu(f,approximate='tanh')
        x=x+linear(act,p+'.mlp.c_proj')
        if collect: cache[i]={'mlp_input':a.detach(),'mlp_act':act.detach()}
    x=ln(x,'transformer.ln_f')
    logits=F.linear(x,state['transformer.wte.weight'])
    return (logits,cache) if collect else logits

def tensor_hash(state:dict)->str:
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
