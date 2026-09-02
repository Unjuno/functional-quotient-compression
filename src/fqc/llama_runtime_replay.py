"""Independent PyTorch replay math for supported Hugging Face Llama layers.

The functions in this module deliberately avoid calling Transformers attention,
RMSNorm, or MLP forward methods. They reconstruct a layer from raw checkpoint
weights plus explicit architecture metadata so a Transformers module can serve
as the reference implementation in a replay witness.

PyTorch is an optional real-model dependency and is imported lazily.
"""
from __future__ import annotations

from typing import Any, Mapping


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in real-model CI
        raise RuntimeError('PyTorch is required for Llama runtime replay') from exc
    return torch


def linear_from_pytorch_out_in(x: Any, weight: Any) -> Any:
    """Apply a raw PyTorch Linear weight `[out,in]` as row-vector `x @ W^T`."""
    return x @ weight.transpose(-1,-2)


def rms_norm(x: Any, weight: Any, eps: float) -> Any:
    """Reproduce Hugging Face LlamaRMSNorm numerics."""
    torch=_torch()
    input_dtype=x.dtype
    y=x.to(torch.float32)
    variance=y.pow(2).mean(-1,keepdim=True)
    y=y*torch.rsqrt(variance+float(eps))
    return weight*y.to(input_dtype)


def rotate_half_split(x: Any) -> Any:
    """Hugging Face Llama `rotate_half`: pair first and second dimension halves."""
    torch=_torch()
    if x.shape[-1]%2:
        raise ValueError('RoPE head dimension must be even')
    half=x.shape[-1]//2
    return torch.cat((-x[...,half:],x[...,:half]),dim=-1)


def default_llama_rope(position_ids: Any, *, head_dim: int, theta: float, dtype: Any, device: Any) -> tuple[Any,Any]:
    """Compute default Llama RoPE cos/sin independently of Transformers."""
    torch=_torch()
    if head_dim<=0 or head_dim%2:
        raise ValueError('head_dim must be positive and even')
    inv_freq=1.0/(float(theta)**(torch.arange(0,head_dim,2,dtype=torch.float32,device=device)/head_dim))
    inv=inv_freq[None,:,None].expand(position_ids.shape[0],-1,1)
    pos=position_ids[:,None,:].to(device=device,dtype=torch.float32)
    freqs=(inv@pos).transpose(1,2)
    emb=torch.cat((freqs,freqs),dim=-1)
    return emb.cos().to(dtype=dtype),emb.sin().to(dtype=dtype)


def apply_rope_half_split(q: Any, k: Any, cos: Any, sin: Any) -> tuple[Any,Any]:
    cos=cos.unsqueeze(1); sin=sin.unsqueeze(1)
    return q*cos+rotate_half_split(q)*sin, k*cos+rotate_half_split(k)*sin


def repeat_kv(x: Any, n_rep: int) -> Any:
    if n_rep<=0:
        raise ValueError('n_rep must be positive')
    if n_rep==1:
        return x
    batch,kv_heads,seqlen,head_dim=x.shape
    return x[:,:,None,:,:].expand(batch,kv_heads,n_rep,seqlen,head_dim).reshape(batch,kv_heads*n_rep,seqlen,head_dim)


def manual_llama_attention(
    hidden_states: Any,
    weights: Mapping[str,Any],
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    attention_mask: Any,
    cos: Any,
    sin: Any,
) -> Any:
    """Reconstruct eager Llama GQA attention from raw projection weights."""
    torch=_torch()
    if q_heads<=0 or kv_heads<=0 or q_heads%kv_heads:
        raise ValueError('q_heads must be a positive multiple of kv_heads')
    batch,seqlen,_=hidden_states.shape
    q=linear_from_pytorch_out_in(hidden_states,weights['WQ']).view(batch,seqlen,q_heads,head_dim).transpose(1,2)
    k=linear_from_pytorch_out_in(hidden_states,weights['WK']).view(batch,seqlen,kv_heads,head_dim).transpose(1,2)
    v=linear_from_pytorch_out_in(hidden_states,weights['WV']).view(batch,seqlen,kv_heads,head_dim).transpose(1,2)
    q,k=apply_rope_half_split(q,k,cos,sin)
    reps=q_heads//kv_heads
    k=repeat_kv(k,reps); v=repeat_kv(v,reps)
    scores=torch.matmul(q,k.transpose(2,3))*(head_dim**-0.5)
    if attention_mask is not None:
        scores=scores+attention_mask[:,:,:,:k.shape[-2]]
    probs=torch.nn.functional.softmax(scores,dim=-1,dtype=torch.float32).to(q.dtype)
    out=torch.matmul(probs,v).transpose(1,2).contiguous().reshape(batch,seqlen,q_heads*head_dim)
    return linear_from_pytorch_out_in(out,weights['WO'])


def manual_llama_mlp(hidden_states: Any, weights: Mapping[str,Any]) -> Any:
    torch=_torch()
    gate=linear_from_pytorch_out_in(hidden_states,weights['gate'])
    up=linear_from_pytorch_out_in(hidden_states,weights['up'])
    return linear_from_pytorch_out_in(torch.nn.functional.silu(gate)*up,weights['down'])


def manual_llama_decoder_layer(
    hidden_states: Any,
    weights: Mapping[str,Any],
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    rms_norm_eps: float,
    attention_mask: Any,
    cos: Any,
    sin: Any,
) -> Any:
    """Reconstruct one pre-norm Llama decoder layer from raw tensors."""
    residual=hidden_states
    x=rms_norm(hidden_states,weights['input_norm'],rms_norm_eps)
    x=manual_llama_attention(
        x,weights,q_heads=q_heads,kv_heads=kv_heads,head_dim=head_dim,
        attention_mask=attention_mask,cos=cos,sin=sin,
    )
    x=residual+x
    residual=x
    x=rms_norm(x,weights['post_attn_norm'],rms_norm_eps)
    x=manual_llama_mlp(x,weights)
    return residual+x
