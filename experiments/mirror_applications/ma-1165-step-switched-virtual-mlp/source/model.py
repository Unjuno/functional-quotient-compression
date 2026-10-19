"""MA-1165: one-parameter-object virtual MLP, step-varying nonlinear Mirror.

Preserves MA-1164 Q^-1*tanh(Q*z+b) residual orbit; dimensions are reduced
only for independent CPU feasibility screening. A single Parameter `u` owns
all learned virtual weights for tied/same/switch/linear modes.
"""
from __future__ import annotations
import io
import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

D, H = 8, 16
N = D*H + H + H*D + D  # 280 virtual MLP coefficients
M = 192
GAMMA = .16


def householder_unit(index: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    j = torch.arange(N, dtype=dtype, device=device)
    vec = torch.sin((.047 + .009 * index) * j + .33 * index) + torch.cos((.067 + .013 * index) * j - .19 * index)
    return vec / vec.norm()


def orthogonal(v: torch.Tensor, step: int, inverse: bool = False, seed: bool = False) -> torch.Tensor:
    """Roll + paired Givens rotation + Householder reflection; exact inverse."""
    if v.shape != (N,):
        raise ValueError(f'expected ({N},), got {tuple(v.shape)}')
    idx = 21 + step if seed else 1 + step
    shift = (7 + idx * 11) % N
    angle = 0.19 + 0.31 * idx
    u = householder_unit(idx, v.dtype, v.device)
    c, s = math.cos(angle), math.sin(angle)
    def givens(z: torch.Tensor, sign: float) -> torch.Tensor:
        x = z.reshape(-1, 2)
        return torch.stack((c*x[:, 0]-sign*s*x[:, 1], sign*s*x[:, 0]+c*x[:, 1]), -1).reshape(-1)
    if not inverse:
        shifted = torch.roll(v, shift)
        rotated = givens(shifted, 1.0)
        return rotated - 2*u*torch.dot(u, rotated)
    refl = v - 2*u*torch.dot(u, v)
    return torch.roll(givens(refl, -1.0), -shift)


def initial_virtual(u: torch.Tensor) -> torch.Tensor:
    if u.ndim != 1 or u.numel() != M:
        raise ValueError('physical vector has incorrect dimension')
    padded = F.pad(u, (0, N-M))
    return orthogonal(padded, 0, seed=True)


def decoder_step(z: torch.Tensor, depth: int, mode: str, gains: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mode in ('tied', 'untied'):
        return z
    if mode == 'gain':
        if gains is None or gains.shape[-1] != 2:
            raise ValueError('expected 2 scalars for gain control')
        # Both affine directions are paid, stage-specific, and learnable.
        position = torch.linspace(-1,1,N,device=z.device,dtype=z.dtype)
        return z*(1 + gains[0]) + gains[1]*position
    level = 0 if mode == 'same' else depth
    oriented = orthogonal(z,level)
    if mode == 'linear':
        response = oriented  # Q^-1 Q z = z; preserves known linear null.
    elif mode in ('same','switch'):
        idx = torch.arange(N, device=z.device, dtype=z.dtype)
        bias = .11*torch.sin(.023*idx + .67*level)
        response = torch.tanh(oriented+bias)
    else:
        raise ValueError('unknown decoder kind')
    return z + GAMMA*orthogonal(response,level,inverse=True)


def mlp_delta(flat:torch.Tensor, x:torch.Tensor)->torch.Tensor:
    w1=flat[:D*H].reshape(H,D)
    b1=flat[D*H:D*H+H]
    w2=flat[D*H+H:D*H+H+H*D].reshape(D,H)
    b2=flat[-D:]
    return F.linear(torch.tanh(F.linear(x,w1,b1)),w2,b2)


class VirtualLoop(nn.Module):
    def __init__(self, mode:str, loops:int, seed:int=0):
        super().__init__()
        if mode not in ('tied','same','switch','linear','gain','untied') or loops not in (2,4):
            raise ValueError('invalid mode or loops')
        self.mode,self.loops,self.seed=mode,loops,seed
        rng=torch.Generator().manual_seed(seed)
        if mode=='untied':
            self.untied=nn.Parameter(torch.randn(loops,N,generator=rng)*.12)
        else:
            self.u=nn.Parameter(torch.randn(M,generator=rng)*.12)
        if mode=='gain':
            self.gains=nn.Parameter(torch.zeros(loops,2))

    def virtual(self)->list[torch.Tensor]:
        if self.mode=='untied':
            return list(self.untied.unbind(0))
        z=initial_virtual(self.u)
        result=[]
        for t in range(self.loops):
            z=decoder_step(z,t,self.mode,self.gains[t] if self.mode=='gain' else None)
            result.append(z)
        return result

    def forward(self,x:torch.Tensor, weight_override:Optional[list[torch.Tensor]]=None,
                return_trace:bool=False):
        if x.ndim !=2 or x.shape[1]!=D:
            raise ValueError('input must be batch x 8')
        weights=self.virtual() if weight_override is None else weight_override
        h=x
        trace=[x]
        for w in weights:
            h=h+(.40/self.loops)*mlp_delta(w,h)
            if return_trace:trace.append(h)
        return (h,trace) if return_trace else h

    def swap_output(self,x:torch.Tensor,order:list[int])->torch.Tensor:
        if sorted(order)!=list(range(self.loops)):raise ValueError('invalid permutation')
        weights=self.virtual()
        return self.forward(x,weight_override=[weights[i] for i in order])

    def payload_bytes(self)->int:
        # Charge parameters, nonlearned index/rules version and full model metadata.
        contents={
            'format':'ma1165-v1', 'mode':self.mode,'loops':self.loops,
            'virtual_coefficients':N,'mirror_program':'seeded-givens-householder-tanh-v1',
            'seed':self.seed,'state':{k:v.detach().cpu().contiguous() for k,v in self.state_dict().items()}
        }
        buf=io.BytesIO();torch.save(contents,buf)
        return len(buf.getvalue())

    def learned_count(self)->int:
        return sum(p.numel() for p in self.parameters())


def make_teacher(kind:str,loops:int,seed:int)->VirtualLoop:
    if kind not in ('aligned','independent'):raise ValueError('unknown teacher')
    model=VirtualLoop('switch' if kind=='aligned' else 'untied',loops,seed=seed+931)
    with torch.no_grad():
        if kind=='aligned':model.u.copy_(torch.randn(M,generator=torch.Generator().manual_seed(seed+284))*0.35)
        else:
            generator=torch.Generator().manual_seed(seed+284)
            model.untied.copy_(torch.randn(loops,N,generator=generator)*.27)
    for p in model.parameters():p.requires_grad_(False)
    return model
