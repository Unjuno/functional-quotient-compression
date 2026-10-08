"""Reduced CoANeRV coordinate-decoder core and token-code families.

This implements the paper's coordinate->token cross-attention interface at a
small CPU scale; it is not the full token former or tokenizer.
"""
from __future__ import annotations
import hashlib, math, subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FRAMES=8
HEIGHT=WIDTH=64
TOKENS=32
TOKEN_DIM=24
TIME_ROLES=8
REGION_ROLES=4
N_HEADS=3
BATCH_COORDS=256


def decode_y4m(path:Path, frames=FRAMES, size=WIDTH):
    """Decode the fixed leading frames with ffmpeg and a deterministic center crop."""
    vf=f"crop=ih:ih:(iw-ih)/2:0,scale={size}:{size}:flags=bicubic"
    cmd=['ffmpeg','-nostdin','-v','error','-i',str(path),'-frames:v',str(frames),'-vf',vf,
         '-fps_mode','passthrough','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    raw=subprocess.check_output(cmd)
    expected=frames*size*size*3
    if len(raw)!=expected:raise ValueError(f'{path}: expected {expected} RGB bytes, got {len(raw)}')
    return torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).copy().reshape(frames,size,size,3)).float()/255.0


def file_sha256(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def all_coords(video:torch.Tensor):
    """Return normalized (x,y,t) coordinates and RGB targets, t in [0,1)."""
    t,h,w,_=video.shape
    ti,yi,xi=torch.meshgrid(torch.arange(t),torch.arange(h),torch.arange(w),indexing='ij')
    coords=torch.stack((xi.float()/(w-1),yi.float()/(h-1),ti.float()/t),dim=-1).reshape(-1,3)
    return coords,video.reshape(-1,3)


def split_indices(n:int,seed:int,train_fraction=.8):
    g=torch.Generator().manual_seed(seed)
    p=torch.randperm(n,generator=g);cut=int(round(n*train_fraction))
    return p[:cut],p[cut:]


class AxisAdaptivePE(nn.Module):
    """Spatial-rich Fourier coordinates, following CoANeRV's anisotropic PE idea."""
    def __init__(self, hidden=TOKEN_DIM):
        super().__init__()
        # 4 spatial and 2 temporal frequencies per axis: 20 inputs.
        self.register_buffer('spatial_freq',torch.exp(torch.linspace(0,math.log(64.0),4)))
        self.register_buffer('temporal_freq',torch.exp(torch.linspace(0,math.log(4.0),2)))
        self.proj=nn.Linear(20,hidden)
    def forward(self,coords):
        x,y,t=coords.unbind(-1)
        sx=x[:,None]*self.spatial_freq[None,:]*math.pi
        sy=y[:,None]*self.spatial_freq[None,:]*math.pi
        st=t[:,None]*self.temporal_freq[None,:]*math.pi
        z=torch.cat((sx.sin(),sx.cos(),sy.sin(),sy.cos(),st.sin(),st.cos()),dim=-1)
        return F.gelu(self.proj(z))


class CoordinateDecoder(nn.Module):
    """Shared coordinate query cross-attention plus RGB readout."""
    def __init__(self,hidden=TOKEN_DIM,heads=N_HEADS,tau=.4):
        super().__init__();self.hidden=hidden;self.heads=heads;self.head_dim=hidden//heads;self.tau=tau
        self.pe=AxisAdaptivePE(hidden)
        self.q=nn.Linear(hidden,hidden,bias=False);self.k=nn.Linear(hidden,hidden,bias=False)
        self.v=nn.Linear(hidden,hidden,bias=False);self.out=nn.Linear(hidden,hidden)
        self.mlp=nn.Sequential(nn.Linear(hidden,48),nn.GELU(),nn.Linear(48,3))
    def forward(self,coords,tokens):
        # coords [B,3], tokens [N,D] (one video/query-bank per minibatch)
        q=self.q(self.pe(coords)).view(-1,self.heads,self.head_dim)
        k=self.k(tokens).view(-1,self.heads,self.head_dim).transpose(0,1)
        v=self.v(tokens).view(-1,self.heads,self.head_dim).transpose(0,1)
        logits=torch.einsum('bhd,hnd->bhn',q,k)*(self.head_dim**-.5/self.tau)
        attn=logits.softmax(dim=-1)
        z=torch.einsum('bhn,hnd->bhd',attn,v).reshape(-1,self.hidden)
        return torch.sigmoid(self.mlp(F.gelu(self.out(z))))


class SharedFactorGenerator(nn.Module):
    """Additive or multiplicative video x time-role x region token generator."""
    def __init__(self,method:str,rank:int,train_videos:int):
        super().__init__();assert method in ('additive','mirror')
        self.method=method;self.rank=rank;self.train_videos=train_videos
        self.video_codes=nn.Parameter(torch.randn(train_videos,rank)*.2)
        self.time=nn.Parameter(torch.randn(TIME_ROLES,rank)*.2)
        self.region=nn.Parameter(torch.randn(REGION_ROLES,rank)*.2)
        self.decode=nn.Parameter(torch.randn(rank,TOKEN_DIM)/math.sqrt(rank))
        self.bias=nn.Parameter(torch.zeros(TOKEN_DIM))
        self.register_buffer('time_index',torch.arange(TOKENS)//REGION_ROLES)
        self.register_buffer('region_index',torch.arange(TOKENS)%REGION_ROLES)
    def table_from_code(self,video_code):
        v=video_code[None,:]
        t=self.time[self.time_index]
        r=self.region[self.region_index]
        if self.method=='mirror':h=v*t*r
        else:h=v+t+r
        return h@self.decode+self.bias
    def train_tables(self):
        return torch.stack([self.table_from_code(c) for c in self.video_codes])
    def shared_state(self):
        return {k:v.detach().cpu() for k,v in self.state_dict().items() if k!='video_codes'}


class PrivateResidual(nn.Module):
    """Rank-2 token-slot residual dictionary added after a Mirror token table."""
    def __init__(self,rank=2):
        super().__init__();self.rank=rank
        self.slot=nn.Parameter(torch.zeros(TOKENS,rank))
    def forward(self,basis):return self.slot@basis


def local_ssim_values(pred,target,window=7):
    """Per-channel local SSIM map, uniform window; caller applies eval mask."""
    x=pred.permute(0,3,1,2);y=target.permute(0,3,1,2)
    mu_x=F.avg_pool2d(x,window,1,window//2);mu_y=F.avg_pool2d(y,window,1,window//2)
    ex2=F.avg_pool2d(x*x,window,1,window//2);ey2=F.avg_pool2d(y*y,window,1,window//2)
    exy=F.avg_pool2d(x*y,window,1,window//2)
    vx=(ex2-mu_x*mu_x).clamp_min(0);vy=(ey2-mu_y*mu_y).clamp_min(0);cov=exy-mu_x*mu_y
    c1=.01**2;c2=.03**2
    score=((2*mu_x*mu_y+c1)*(2*cov+c2))/((mu_x.square()+mu_y.square()+c1)*(vx+vy+c2)+1e-12)
    return score.mean(dim=1)


def evaluate(decoder,tokens,video,eval_mask,batch=512):
    coords,targets=all_coords(video);device=next(decoder.parameters()).device
    preds=[];start=torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    import time
    t0=time.perf_counter()
    with torch.inference_mode():
        for ix in range(0,len(coords),batch):preds.append(decoder(coords[ix:ix+batch].to(device),tokens.to(device)).cpu())
    wall=time.perf_counter()-t0;pred=torch.cat(preds).reshape_as(video);mask=eval_mask.reshape_as(video[...,0])
    sq=(pred-video).square().mean(dim=-1)
    mse=sq[mask].mean().item();psnr=10*math.log10(1.0/max(mse,1e-12))
    ssim_map=local_ssim_values(pred,video)
    ssim=float(ssim_map[mask].mean())
    return {'heldout_mse':mse,'heldout_psnr_db':psnr,'heldout_ssim':ssim,
            'eval_coordinates':int(mask.sum()),'decoder_eval_wall_s':wall,
            'query_pixels_per_second':float(len(coords))/max(wall,1e-9)}
