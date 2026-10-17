"""MA-1162: multi-head logical Mirror views over ONE canonical physical KV.

Each logical head shares physical q/k/v projections. For linear orthogonal
right-views, key and value rotations are moved onto current query/readout.
Current-query/post-attention nonlinear sandwiches preserve canonical KV, but
pre-attention nonlinear VALUE sandwiches cannot generally be folded.
"""
import io
import math
import torch
from torch import nn


def rotate_pairs(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Same independent 2-plane rotation angle for every adjacent pair.

    Broadcast angle across all non-feature dimensions, including any heads.
    """
    if x.shape[-1] % 2:
        raise ValueError('even head dimension required')
    c, s = torch.cos(a), torch.sin(a)
    z0, z1 = x[..., ::2], x[..., 1::2]
    # torch.stack interleaves pair coordinates without transposed copies.
    return torch.stack((c*z0-s*z1,s*z0+c*z1),dim=-1).flatten(-2)


def attention(q,k,v):
    return torch.softmax((q @ k.transpose(-1,-2))/math.sqrt(q.shape[-1]),dim=-1) @ v


def attention_weights(q,k):
    return torch.softmax((q @ k.transpose(-1,-2))/math.sqrt(q.shape[-1]),dim=-1)


class SharedMultihead(nn.Module):
    """Multihead mirror with 1 physical K/V store, 2*heads trainable View angles.

    Mode linear: static orthogonal directions, strictly foldable.
    Mode nonlinear: nonlinear query and post-attention output, still cache-safe.
    Mode gain: same-count non-Mirror multiplicative gain control.
    Mode null: all heads are identical, no head-specific function.
    """
    def __init__(self,d_model=16,d_head=8,heads=4,mode='linear'):
        super().__init__()
        if mode not in ('linear','nonlinear','gain','null'):
            raise ValueError(mode)
        self.d_model,self.d_head,self.heads,self.mode=d_model,d_head,heads,mode
        # Freeze shared QKV for isolated code optimization. Count stored bytes.
        self.register_buffer('Wq',torch.randn(d_model,d_head)/math.sqrt(d_model))
        self.register_buffer('Wk',torch.randn(d_model,d_head)/math.sqrt(d_model))
        self.register_buffer('Wv',torch.randn(d_model,d_head)/math.sqrt(d_model))
        if mode!='null':
            self.q_code=nn.Parameter(torch.zeros(heads))
            self.v_code=nn.Parameter(torch.zeros(heads))

    def forward(self,x,return_attention=False):
        """Full causal batch forward, returns [B,T,H,dhead] logical head results."""
        q0,k0,v0=x@self.Wq,x@self.Wk,x@self.Wv
        T=x.shape[1]
        mask=torch.ones(T,T,device=x.device,dtype=torch.bool).triu(1)
        outs,atts=[],[]
        for j in range(self.heads):
            if self.mode=='null':
                q=q0
            elif self.mode=='linear':
                q=rotate_pairs(q0,self.q_code[j])
            elif self.mode=='nonlinear':
                q=rotate_pairs(torch.tanh(rotate_pairs(q0,self.q_code[j])),-self.q_code[j])
            else:
                q=q0*torch.exp(self.q_code[j])
            a=(q@k0.transpose(-1,-2)/math.sqrt(self.d_head)).masked_fill(mask,-torch.inf).softmax(-1)
            v=a@v0
            if self.mode=='linear':
                v=rotate_pairs(v,self.v_code[j])
            elif self.mode=='nonlinear':
                v=rotate_pairs(torch.tanh(rotate_pairs(v,self.v_code[j])),-self.v_code[j])
            elif self.mode=='gain':
                v=v*torch.exp(self.v_code[j])
            outs.append(v)
            atts.append(a)
        out=torch.stack(outs,dim=2)
        return (out,torch.stack(atts,dim=2)) if return_attention else out

    def decode_lazy(self,q0,k0,v0):
        outs=[]
        for j in range(self.heads):
            if self.mode=='linear':
                q=rotate_pairs(q0,self.q_code[j])
            elif self.mode=='nonlinear':
                q=rotate_pairs(torch.tanh(rotate_pairs(q0,self.q_code[j])),-self.q_code[j])
            elif self.mode=='gain':
                q=q0*torch.exp(self.q_code[j])
            else:q=q0
            v=attention(q,k0,v0)
            if self.mode=='linear': v=rotate_pairs(v,self.v_code[j])
            elif self.mode=='nonlinear': v=rotate_pairs(torch.tanh(rotate_pairs(v,self.v_code[j])),-self.v_code[j])
            elif self.mode=='gain':v=v*torch.exp(self.v_code[j])
            outs.append(v)
        return torch.stack(outs,dim=2)

    def decode_materialized(self,q0,k0,v0):
        """Only equivalent for linear Views; duplicates full prefix for each head."""
        if self.mode!='linear':raise ValueError('materialized reference tests only linear')
        outs=[]
        for j in range(self.heads):
            # Choose q-angle = relative angle and k-angle 0; function is identical.
            q=rotate_pairs(q0,self.q_code[j])
            v=attention(q,k0,rotate_pairs(v0,self.v_code[j]))
            outs.append(v)
        return torch.stack(outs,dim=2)

    def decode_folded_native(self,query_x,k0,v0):
        """Precompile per-head q projection and post-attention value operation.

        Fixed angles only; physical canonical K/V are shared. The output-space
        linear rotation may be pre-absorbed into a later output projection.
        """
        if self.mode!='linear': raise ValueError('fixed linear only')
        outs=[]
        for j in range(self.heads):
            # angle-dependent Wq_j is paid as one full projection per head in
            # this materialized folded-native implementation.
            p=rotate_pairs(torch.eye(self.d_head,device=query_x.device,dtype=query_x.dtype),self.q_code[j])
            q=query_x@(self.Wq@p)
            out=attention(q,k0,v0)
            # Output rotations are fixed, but we retain the final coordinate
            # operation to preserve per-head output identity in this harness.
            outs.append(rotate_pairs(out,self.v_code[j]))
        return torch.stack(outs,dim=2)

    def canonical_bytes(self,prefix_len,batch=1,dtype_bytes=4):
        return 2*batch*prefix_len*self.d_head*dtype_bytes

    def serialized_bytes(self):
        f=io.BytesIO()
        torch.save({'cfg':{'d_model':self.d_model,'d_head':self.d_head,
              'heads':self.heads,'mode':self.mode},'state':self.state_dict()},f)
        return len(f.getvalue())


def code_pairwise_kl(att):
    """Mean pairwise KL divergence of attention distributions across heads."""
    # att [B,T,H,T]; mean of ordered pairs, evaluate proper masked zeros.
    H=att.shape[2]
    if H<2:return 0.
    vals=[]
    for j in range(H):
        for k in range(j+1,H):
            p,q=att[:,:,j,:],att[:,:,k,:]
            vals.append((p*(p.clamp_min(1e-20).log()-q.clamp_min(1e-20).log())).sum(-1).mean())
    return torch.stack(vals).mean().item()
