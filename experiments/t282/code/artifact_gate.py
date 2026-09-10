"""Semantic model gate for the supported GPT-Neo FQCWR001 family.
A checksum is an integrity check, not proof that tensor names/shapes match config.
This wrapper keeps the legacy decoder unchanged and rejects semantic mismatches.
It is not a claim of a general security audit or a production-safe parser.
"""
from __future__ import annotations
import math
from independent_decoder import decode_model

def expected_shapes(cfg):
    def positive(name, fallback=None):
        v=cfg.get(name,fallback)
        if type(v) is not int or v<=0: raise ValueError(f'Invalid config {name}')
        return v
    d=positive('hidden_size'); l=positive('num_layers'); v=positive('vocab_size'); p=positive('max_position_embeddings')
    h=positive('intermediate_size',4*d) if cfg.get('intermediate_size') is not None else 4*d
    heads=positive('num_heads')
    if d%heads: raise ValueError('hidden/head mismatch')
    if cfg.get('activation_function')!='gelu_new':raise ValueError('Unsupported activation')
    a=cfg.get('attention_layers')
    if not isinstance(a,list) or len(a)!=l or any(x not in ('local','global') for x in a):raise ValueError('Attention layer schema')
    if not isinstance(cfg.get('layer_norm_epsilon'),(int,float)) or not 0<cfg['layer_norm_epsilon']<1:raise ValueError('Layer norm epsilon')
    if 'local' in a:positive('window_size')
    result={'transformer.wte.weight':(v,d),'transformer.wpe.weight':(p,d),
            'transformer.ln_f.weight':(d,),'transformer.ln_f.bias':(d,)}
    for i in range(l):
        base=f'transformer.h.{i}'
        for norm in ('ln_1','ln_2'):
            for kind in ('weight','bias'):result[f'{base}.{norm}.{kind}']=(d,)
        for name in ('q','k','v','out'):result[f'{base}.attn.attention.{name}_proj.weight']=(d,d)
        result[f'{base}.attn.attention.out_proj.bias']=(d,)
        result[f'{base}.mlp.c_fc.weight']=(h,d);result[f'{base}.mlp.c_fc.bias']=(h,)
        result[f'{base}.mlp.c_proj.weight']=(d,h);result[f'{base}.mlp.c_proj.bias']=(d,)
    return result

def checked_decode(path,device='cpu'):
    cfg,state,metadata=decode_model(path,device)
    expected=expected_shapes(cfg)
    if set(state)!=set(expected):raise ValueError('Tensor names do not match config')
    for n,shape in expected.items():
        if tuple(state[n].shape)!=shape:raise ValueError(f'Tensor shape/config mismatch: {n}')
    paid=sum(math.prod(s) for s in expected.values())
    if paid!=sum(t.numel() for t in state.values()):raise ValueError('Paid count mismatch')
    return cfg,state,metadata
