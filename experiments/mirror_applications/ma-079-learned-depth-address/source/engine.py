"""Depth address extrapolation screen for MA-079."""
import io, math, time
from dataclasses import dataclass
import torch
from torch import nn

DEPTH, WIDTH, TRAIN_N, TEST_N = 8, 8, 256, 128
SEEN = (0, 2, 5, 7)
HOLDOUT = (1, 3, 4, 6)
METHODS = ("tied", "static_lora", "generated_address", "mirror_address", "untied")

def rot(theta):
    c,s=torch.cos(theta),torch.sin(theta); r=torch.eye(WIDTH)
    r[0,0],r[0,1]=c,-s; r[1,0],r[1,1]=s,c
    return r

def zvals(): return torch.linspace(-1.,1.,DEPTH)

def make_world(seed, aligned):
    g=torch.Generator().manual_seed(seed)
    x=torch.randn(TRAIN_N,WIDTH,generator=g); xt=torch.randn(TEST_N,WIDTH,generator=g)
    base=torch.randn(WIDTH,WIDTH,generator=g)/math.sqrt(WIDTH)
    if aligned:
        zs=zvals(); mats=torch.stack([rot(0.7*z+0.25*z*z) @ base @ rot(-0.7*z-0.25*z*z) for z in zs])
    else: mats=torch.randn(DEPTH,WIDTH,WIDTH,generator=g)/math.sqrt(WIDTH)
    return x,torch.einsum('dij,nj->dni',mats,x),xt,torch.einsum('dij,nj->dni',mats,xt)

class AddressModel(nn.Module):
    def __init__(self,method):
        super().__init__(); self.method=method
        if method=='untied':
            self.weight=nn.Parameter(torch.empty(DEPTH,WIDTH,WIDTH)); nn.init.normal_(self.weight,std=1/math.sqrt(WIDTH))
        else:
            self.weight=nn.Parameter(torch.empty(WIDTH,WIDTH)); nn.init.normal_(self.weight,std=1/math.sqrt(WIDTH))
        if method=='static_lora':
            self.a=nn.Parameter(torch.randn(DEPTH,WIDTH,1)*.05); self.b=nn.Parameter(torch.zeros(DEPTH,1,WIDTH))
        if method=='generated_address': self.gain=nn.Parameter(torch.zeros(2,WIDTH))
        if method=='mirror_address': self.angle_coeff=nn.Parameter(torch.zeros(2))
    def matrices(self):
        zs=zvals(); feats=torch.stack((zs,zs.square()),dim=-1)
        if self.method=='untied': return self.weight
        if self.method=='static_lora': return self.weight[None]+torch.bmm(self.a,self.b)
        if self.method=='generated_address':
            gains=1.+feats @ self.gain
            return gains[:,:,None]*self.weight[None]
        if self.method=='mirror_address':
            angles=feats @ self.angle_coeff
            return torch.stack([rot(t) @ self.weight @ rot(-t) for t in angles])
        return self.weight[None].expand(DEPTH,-1,-1)
    def forward(self,x): return torch.einsum('dij,nj->dni',self.matrices(),x)

@dataclass
class Result:
    method:str; seed:int; condition:str; mse:float; holdout_mse:float; seen_mse:float
    payload:int; updates:int; examples:int; macs:int; wall:float; throughput:float

def payload_bytes(model):
    b=io.BytesIO(); torch.save({'method':model.method,'state_dict':model.state_dict()},b); return b.getvalue()

def train_one(method,seed,condition,updates=600,lr=.01):
    torch.manual_seed(seed+223); aligned=condition=='aligned'; x,y,xt,yt=make_world(seed,aligned)
    model=AddressModel(method); opt=torch.optim.Adam(model.parameters(),lr=lr); start=time.perf_counter()
    for _ in range(updates):
        pred=model(x)[list(SEEN)]; loss=(pred-y[list(SEEN)]).square().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    wall=time.perf_counter()-start
    with torch.no_grad():
        diff=model(xt)-yt; per=diff.square().mean(dim=(1,2))
        mse=float(diff.square().mean()); held=float(per[list(HOLDOUT)].mean()); seen=float(per[list(SEEN)].mean())
    macs=updates*TRAIN_N*len(SEEN)*WIDTH*WIDTH*3
    if method=='mirror_address': macs+=updates*3*2*DEPTH*WIDTH**3
    elif method=='generated_address': macs+=updates*DEPTH*WIDTH*2
    elif method=='static_lora': macs+=updates*3*2*DEPTH*WIDTH
    return Result(method,seed,condition,mse,held,seen,len(payload_bytes(model)),updates,updates*TRAIN_N*len(SEEN),int(macs),wall,updates*TRAIN_N*len(SEEN)/wall)

def run(seed,condition,updates=600,lr=.01): return [train_one(m,seed,condition,updates,lr) for m in METHODS]
