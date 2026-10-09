"""Nonlinear expert bank with optional Mirror views and private output residuals."""
import torch
from torch import nn
from torch.nn import functional as F

D_IN,HIDDEN,D_OUT,ROLES=16,32,16,4
METHODS=("independent","tied","tied_rank1","tied_rank2","mirror","mirror_rank1","mirror_rank2")

def rotation(theta,dim,transpose=False):
    c,s=torch.cos(theta),torch.sin(theta)
    if transpose:s=-s
    r=torch.eye(dim,dtype=theta.dtype,device=theta.device).expand(len(theta),dim,dim).clone()
    r[:,0,0]=c;r[:,0,1]=-s;r[:,1,0]=s;r[:,1,1]=c
    return r

class ExpertBank(nn.Module):
    def __init__(self,method,seed):
        super().__init__();self.method=method
        if method not in METHODS:raise ValueError(method)
        torch.manual_seed(seed)
        if method=='independent':
            self.win=nn.Parameter(torch.randn(ROLES,HIDDEN,D_IN)*.12)
            self.wout=nn.Parameter(torch.randn(ROLES,D_OUT,HIDDEN)*.12)
            rank=0
        else:
            self.win=nn.Parameter(torch.randn(HIDDEN,D_IN)*.12)
            self.wout=nn.Parameter(torch.randn(D_OUT,HIDDEN)*.12)
            rank= int(method[-1]) if 'rank' in method else 0
            if rank:
                self.left=nn.Parameter(torch.randn(ROLES,D_OUT,rank)*.02)
                self.right=nn.Parameter(torch.randn(ROLES,rank,HIDDEN)*.02)
            if method.startswith('mirror'):
                self.input_angle=nn.Parameter(torch.zeros(ROLES))
                self.output_angle=nn.Parameter(torch.zeros(ROLES))
        self.rank=rank

    def matrices(self):
        if self.method=='independent':return self.win,self.wout
        return self.win.unsqueeze(0).expand(ROLES,-1,-1),self.wout.unsqueeze(0).expand(ROLES,-1,-1)

    def forward(self,x,roles):
        win,wout=self.matrices();wi=win[roles];wo=wout[roles]
        h_in=x
        if self.method.startswith('mirror'):
            rin=rotation(self.input_angle, D_IN, transpose=True)[roles]
            h_in=torch.bmm(rin,x.unsqueeze(-1)).squeeze(-1)
        h=F.gelu(torch.bmm(wi,h_in.unsqueeze(-1)).squeeze(-1))
        y=torch.bmm(wo,h.unsqueeze(-1)).squeeze(-1)
        if self.rank:
            y=y+torch.bmm(self.left[roles],torch.bmm(self.right[roles],h.unsqueeze(-1))).squeeze(-1)
        if self.method.startswith('mirror'):
            rout=rotation(self.output_angle,D_OUT)[roles]
            y=torch.bmm(rout,y.unsqueeze(-1)).squeeze(-1)
        return y

