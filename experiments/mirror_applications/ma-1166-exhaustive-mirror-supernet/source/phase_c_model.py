"""MA-1166 Phase C: literal 63-edge no-prune Mirror supergraph and strong controls.
All model variables are dimensionless. Training is performed outside this module.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

D=6
H=8
VIRTUAL= D*H+H+H*D+D
PHYSICAL=64+63+63*2


def householder(x,unit):
    return x-2*(x*unit).sum(dim=-1,keepdim=True)*unit


class GraphSpec:
    def __init__(self):
        self.edges=tuple((i,j,k) for j in range(1,7) for i in range(j) for k in range(3))
        assert len(self.edges)==63
        self.dst={j:torch.tensor([q for q,e in enumerate(self.edges) if e[1]==j],dtype=torch.long) for j in range(1,7)}
        self.index={e:q for q,e in enumerate(self.edges)}


class VirtualSupernet(nn.Module):
    """One trainable nn.Parameter of 253 scalars; all 63 addresses stay executable.

    decoder='mirror': nonlinear weight-space Householder/tanh/inverse-Householder
    decoder='affine': direct low-rank/tanh shared basis with equally many learned codes.
    """
    def __init__(self,decoder='mirror',seed=1977):
        super().__init__()
        assert decoder in ('mirror','affine')
        self.spec=GraphSpec()
        self.decoder=decoder
        self.seed=int(seed)
        g=torch.Generator(device='cpu').manual_seed(self.seed)
        rv=lambda *dims: torch.randn(*dims,generator=g,dtype=torch.float32)
        self.register_buffer('embedding',rv(64,110)/math.sqrt(64),persistent=False)
        self.register_buffer('fixed_edge',rv(63,110)*.12,persistent=False)
        self.register_buffer('fixed_bias',rv(63,110)*.28,persistent=False)
        unit=lambda z: F.normalize(z,dim=-1,eps=1e-8)
        for name in ('v1','v2','v1shift','v2shift','a1','a2'):
            self.register_buffer(name,unit(rv(63,110)),persistent=False)
        self.theta=nn.Parameter(torch.zeros(PHYSICAL))
        with torch.no_grad():
            self.theta[:64].copy_(rv(64)*.62)
            self.theta[64:127].fill_(-.15)
            self.theta[127:].copy_(rv(126)*.18)

    def decode(self,idx):
        # z and all view matrices use virtual-weight coordinates, NOT MLP activations.
        core=self.theta[:64]
        codes=self.theta[127:].reshape(63,2)[idx]
        z=torch.matmul(core,self.embedding).unsqueeze(0)+self.fixed_edge[idx]
        if self.decoder=='mirror':
            a=F.normalize(self.v1[idx]+codes[:,0:1]*self.v1shift[idx],dim=-1)
            b=F.normalize(self.v2[idx]+codes[:,1:2]*self.v2shift[idx],dim=-1)
            rotated=householder(householder(z,a),b)
            warped=torch.tanh(rotated*2.2+self.fixed_bias[idx])
            return z+.65*householder(householder(warped,b),a)
        return z+.65*torch.tanh(z*2.2+self.fixed_bias[idx])+codes[:,0:1]*self.a1[idx]+codes[:,1:2]*self.a2[idx]

    @staticmethod
    def _edge_result(inputs,w):
        k=len(w)
        a=w[:,:48].reshape(k,6,8)*.62
        b=w[:,48:56]*.27
        c=w[:,56:104].reshape(k,8,6)*.62
        d=w[:,104:110]*.27
        return torch.bmm(torch.tanh(torch.bmm(inputs,a)+b[:,None,:]),c)+d[:,None,:]

    def forward(self,x,sample_probability=1.,return_stats=False):
        assert 0<sample_probability<=1
        states=[x]
        count=0
        for j in range(1,7):
            idx=self.spec.dst[j]
            if sample_probability<1.:
                selected=torch.rand(len(idx),device=x.device)<sample_probability
                if not selected.any():
                    selected[torch.randint(len(idx),(1,),device=x.device)]=True
                idx=idx[selected]
            count+=len(idx)
            w=self.decode(idx)
            ins=torch.stack([states[self.spec.edges[q][0]] for q in idx.tolist()],dim=0)
            out=self._edge_result(ins,w)
            gate=torch.sigmoid(self.theta[64:127][idx]).view(-1,1,1)
            weight=gate/sample_probability
            delta=(weight*out).sum(dim=0)*(1.2/(j*3))
            states.append(states[-1]+delta)
        if return_stats:
            return states[-1],dict(edge_evals=count,candidate_count=63,sample_probability=sample_probability)
        return states[-1]


class NativeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1=nn.Linear(6,19)
        self.w2=nn.Linear(19,6)
    def forward(self,x):
        return x+1.2*self.w2(torch.tanh(self.w1(x)))


class NativeSixBlocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=nn.ModuleList([nn.Sequential(nn.Linear(6,8),nn.Tanh(),nn.Linear(8,6)) for _ in range(6)])
    def forward(self,x):
        for block in self.blocks:
            x=x+.42*block(x)
        return x


class Teacher:
    def __init__(self,world,seed):
        assert world in ('aligned','independent')
        self.world=world
        self.seed=int(seed)
        self.spec=GraphSpec()
        self.active_ids=[]
        for j in range(1,7):
            self.active_ids.append(self.spec.index[(0,j,j%3)])
            if j>1:
                self.active_ids.append(self.spec.index[(j-1,j,(j+1)%3)])
        self.active_by_dst={j:torch.tensor([q for q in self.active_ids if self.spec.edges[q][1]==j]) for j in range(1,7)}
        if world=='aligned':
            self.aligned=VirtualSupernet('mirror',seed=1977)
            g=torch.Generator().manual_seed(seed+9000)
            with torch.no_grad():
                self.aligned.theta[:64].copy_(torch.randn(64,generator=g)*1.5)
                self.aligned.theta[127:].copy_(torch.randn(126,generator=g)*.55)
        else:
            g=torch.Generator().manual_seed(seed+19000)
            self.independent=torch.randn(63,110,generator=g)*1.35

    @torch.no_grad()
    def __call__(self,x):
        states=[x]
        for j in range(1,7):
            idx=self.active_by_dst[j]
            ins=torch.stack([states[self.spec.edges[q][0]] for q in idx.tolist()],dim=0)
            w=self.aligned.decode(idx) if self.world=='aligned' else self.independent[idx]
            out=VirtualSupernet._edge_result(ins,w)
            # Relative to 63-edge supergraph, teacher assigns nonzero value to
            # a small number of long/short routes; teacher weights are not revealed.
            states.append(states[-1]+(out.sum(0)*(1.2/(j*3))))
        return states[-1]
