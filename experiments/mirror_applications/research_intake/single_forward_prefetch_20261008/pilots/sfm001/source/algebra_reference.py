#!/usr/bin/env python3
"""SFM001 lightweight algebra-only reproduction. CPU torch 2.10 float64.
The full benchmark/source data are in the frozen conversation artifact ZIP.
No training, LLM or GPU speedup claim is made by this reference preflight.
"""
import json
import math
import torch
import torch.nn.functional as F

SEEDS=(101,102,103,104,105)
torch.set_num_threads(1)


def rotate(v,a,inverse=False):
    """Pairwise Givens on final axis; broadcast shape permitted."""
    b=v.reshape(*v.shape[:-1],v.shape[-1]//2,2)
    c,s=torch.cos(a),torch.sin(a)*(-1.0 if inverse else 1.0)
    r=torch.stack((c*b[...,0]-s*b[...,1],s*b[...,0]+c*b[...,1]),-1)
    return r.reshape(*r.shape[:-2],v.shape[-1])


def full(x,U,D,angles):
    return torch.stack([rotate(F.gelu(rotate(x@U,a)),a,True)@D for a in angles])


def cached(x,U,D,angles):
    z=x@U
    return torch.stack([rotate(F.gelu(rotate(z,a)),a,True)@D for a in angles])


def run(seed):
    g=torch.Generator().manual_seed(seed)
    rn=lambda shape:torch.randn(shape,generator=g,dtype=torch.float64)
    x=rn((23,16));U=rn((16,32))/4;D=rn((32,16))/math.sqrt(32)
    angles=(torch.rand((5,16),generator=g,dtype=torch.float64)*2-1)*0.8
    output_angles=angles[:,:8]
    a,b=full(x,U,D,angles),cached(x,U,D,angles)
    exact=float((a-b).abs().max())
    y0=F.gelu(x@U)@D
    output_direct=torch.stack([rotate(F.gelu(x@U)@D,a) for a in output_angles])
    output_cheap=rotate(y0.unsqueeze(0),output_angles[:,None,:])
    output_error=float((output_direct-output_cheap).abs().max())
    z=x@U;perm=torch.randperm(32,generator=g)
    recovered=torch.empty_like(z);recovered[:,perm]=F.gelu(z[:,perm])
    permutation_error=float((recovered-F.gelu(z)).abs().max())
    K=rn((64,16));V=rn((64,16));Q=rn((7,16));theta=angles[0,:8]
    P=(Q@K.T/4).softmax(-1)
    value_error=float(((P@rotate(V,theta))-rotate(P@V,theta)).abs().max())
    key_only=float(((Q@K.T)-(Q@rotate(K,theta).T)).abs().max())
    coupled=float(((Q@K.T)-(rotate(Q,theta)@rotate(K,theta).T)).abs().max())
    pair=torch.tensor([[1.25,-0.48],[-0.48,1.25]],dtype=torch.float64)
    native_pair=F.gelu(pair).sum(-1)
    theta2=torch.tensor([0.67],dtype=torch.float64)
    mirrored_pair=rotate(F.gelu(rotate(pair,theta2)),theta2,True).sum(-1)
    gap=float((mirrored_pair[0]-mirrored_pair[1]).abs())
    assert float((native_pair[0]-native_pair[1]).abs())<1e-12
    U1=U.clone().requires_grad_();D1=D.clone().requires_grad_()
    gu1,gd1=torch.autograd.grad(full(x,U1,D1,angles).square().mean(),(U1,D1))
    U2=U.clone().requires_grad_();D2=D.clone().requires_grad_()
    gu2,gd2=torch.autograd.grad(cached(x,U2,D2,angles).square().mean(),(U2,D2))
    grad=max(float((gu1-gu2).abs().max()),float((gd1-gd2).abs().max()))
    for e in (exact,output_error,permutation_error,value_error,coupled,grad):
        assert e<1e-10,(seed,e)
    assert key_only>0.01 and gap>0.01
    return dict(seed=seed,hidden_exact=exact,output_exact=output_error,
                attention_value_exact=value_error,attention_coupled_score_exact=coupled,
                attention_key_only_score_change=key_only,
                gradient_exact=grad,base_output_collision_mirror_gap=gap,
                output_only_min_uniform_error=gap/2,all_pass=True)


def simulated_prefetch(M=4,mb=16,gbps=12,n=12):
    # Hypothetical 2-buffer transfer (not actual measured CPU-GPU DMA).
    transfer_s=0.00025+mb*1024**2/(gbps*1e9)
    compute_s=(1.0+0.18*M)/1e3
    critical_s=transfer_s+n*compute_s+(n-1)*max(0,transfer_s-compute_s)
    ondemand_s=n*(transfer_s+compute_s)
    assert 0<critical_s<=ondemand_s
    return dict(views=M,assumed_block_mib=mb,assumed_bandwidth_gbps=gbps,
                transfer_s=transfer_s,per_group_compute_s=compute_s,
                oracle_pregated_wall_s=critical_s,ondemand_wall_s=ondemand_s,
                ordinary_factorized_equal_wall_s=critical_s,real_DMA_measured=False)


if __name__=='__main__':
    print(json.dumps(dict(worlds=[run(s) for s in SEEDS],overlap=simulated_prefetch()),indent=2))
