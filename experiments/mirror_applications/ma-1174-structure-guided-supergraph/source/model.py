"""MA-1174: virtual Mirror supergraph regression mechanism (synthetic, not natural LM)."""
from __future__ import annotations
import io, math
import torch
from torch import nn
from torch.nn import functional as F

D, NODES, VIEWS, EDGES, PAGES, LOOPS = 8, 4, 2, 32, 4, 2


def fixed_program(seed=1174):
    gen=torch.Generator().manual_seed(seed)
    w=2*D*D
    perms=torch.stack([torch.randperm(w,generator=gen) for _ in range(PAGES)])
    signs=torch.stack([torch.randint(0,2,(w,),generator=gen).float()*2-1 for _ in range(PAGES)])
    u=torch.randn(D,generator=gen);u/=u.norm()
    house=torch.eye(D)-2*torch.outer(u,u)
    mats=[]
    for _ in range(NODES):
        a=torch.randn(D,D,generator=gen)
        q,r=torch.linalg.qr(a)
        mats.append(q*torch.sign(torch.diag(r))[None,:])
    bases=torch.stack(mats)
    src=[];dst=[]
    for node in range(NODES):
        for source in range(NODES):
            for _view in range(VIEWS):
                src.append(source);dst.append(node)
    return perms,signs,house,bases,torch.tensor(src),torch.tensor(dst)

class VirtualGraph(nn.Module):
    def __init__(self,kind="mirror"):
        super().__init__()
        assert kind in ("mirror","direct_tanh","affine")
        self.kind=kind
        self.core=nn.Parameter(0.45*torch.randn(2,D,D))
        self.address=nn.Parameter(0.12*torch.randn(EDGES,PAGES))
        self.gates=nn.Parameter(torch.full((EDGES,),-0.7))
        self.bias=nn.Parameter(0.03*torch.randn(NODES,D))
        self.register_buffer('active',torch.ones(EDGES,dtype=torch.bool))
        perms,signs,house,bases,src,dst=fixed_program()
        for name,val in [('perms',perms),('signs',signs),('house',house),('bases',bases),('src',src),('dst',dst)]:
            self.register_buffer(name,val,persistent=False)

    def weights(self,indices=None):
        if indices is None:indices=self.active.nonzero().flatten()
        if indices.numel()==0:return self.core.new_zeros((0,2,D,D))
        pages=self.core.flatten()[self.perms]*self.signs
        w=(F.softmax(self.address[indices]*2,dim=-1)@pages).reshape(-1,2,D,D)
        if self.kind=='mirror':
            h=self.house
            w=h @ (torch.tanh(2*(h @ w @ h))/2) @ h
        elif self.kind=='direct_tanh':w=torch.tanh(2*w)/2
        return w

    def forward(self,x,mask=None):
        if x.ndim!=2 or x.shape[-1]!=D:raise ValueError('expected [batch,8]')
        a=self.active if mask is None else mask
        ids=a.nonzero().flatten()
        state0=torch.einsum('bd,ndh->bnh',x,self.bases)+self.bias[None]
        state=state0
        if ids.numel()>0:
            w=self.weights(ids)
            src=self.src[ids];dst=self.dst[ids]
            gate=torch.sigmoid(self.gates[ids])
            for _ in range(LOOPS):
                z=state[:,src,:]
                h=F.gelu(torch.einsum('bei,eij->bej',z,w[:,0]))
                y=torch.einsum('bei,eij->bej',h,w[:,1]) * gate[None,:,None]
                msg=state.new_zeros(state.shape).index_add(1,dst,y)
                state=state+0.8*msg/math.sqrt(NODES*VIEWS)
        delta=state-state0
        return delta.mean(1)+0.6*delta[:,0,:]

    def cut(self,ids):
        for i in ids:
            if not bool(self.active[i]):raise ValueError('inactive edge')
            self.active[i]=False
    def active_count(self):return int(self.active.sum().item())
    def packet(self):
        ids=self.active.nonzero().flatten()
        return dict(kind=self.kind,core=self.core.detach().cpu(),address=self.address.detach().cpu()[ids].contiguous(),gates=self.gates.detach().cpu()[ids].contiguous(),bias=self.bias.detach().cpu(),ids=ids.to(torch.int16).cpu(),program_seed=1174,nodes=NODES,views=VIEWS,loops=LOOPS)
    def packet_bytes(self):
        b=io.BytesIO();torch.save(self.packet(),b);return len(b.getvalue())
    def reload_packet(self,p):
        if p['kind']!=self.kind:raise ValueError('wrong kind')
        with torch.no_grad():
            self.core.copy_(p['core']);self.bias.copy_(p['bias']);self.gates.zero_();self.address.zero_()
            self.active.zero_()
            ids=p['ids'].long();self.active[ids]=True
            self.gates[ids]=p['gates'];self.address[ids]=p['address']

class DenseControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(D,18);self.fc2=nn.Linear(18,D)
    def forward(self,x):return self.fc2(F.gelu(self.fc1(x)))
    def packet_bytes(self):
        b=io.BytesIO();torch.save({k:v.detach().cpu() for k,v in self.state_dict().items()},b);return len(b.getvalue())


def new_model(kind,seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return DenseControl() if kind=='native' else VirtualGraph(kind)
