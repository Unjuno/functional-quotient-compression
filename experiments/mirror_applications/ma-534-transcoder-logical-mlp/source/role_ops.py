"""Role assignment, sparse decoding and orthogonal feature Views for MA-534."""
import torch

def givens_view(code: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    pairs=code.reshape(*code.shape[:-1],-1,2)
    c=angles.cos().unsqueeze(0) if code.ndim==2 else angles.cos()
    s=angles.sin().unsqueeze(0) if code.ndim==2 else angles.sin()
    a,b=pairs[...,0],pairs[...,1]
    out=torch.stack([c*a-s*b,s*a+c*b],dim=-1)
    return out.reshape_as(code)

def sparse_decode(code: torch.Tensor, decoder: torch.Tensor, bias: torch.Tensor, skip: torch.Tensor, x: torch.Tensor, k: int):
    vals,idx=code.topk(k,dim=-1)
    y=(vals.unsqueeze(-1)*decoder[idx]).sum(dim=1)+bias+x@skip.T
    return y,idx,vals

def fit_centroids(x: torch.Tensor, k: int, seed: int, iterations: int=50):
    g=torch.Generator(device=x.device); g.manual_seed(seed)
    idx=torch.randperm(len(x),generator=g,device=x.device)[:k]
    c=x[idx].clone()
    for _ in range(iterations):
        labels=torch.cdist(x,c).argmin(dim=1)
        for r in range(k):
            m=labels==r
            if m.any(): c[r]=x[m].mean(0)
    return c,torch.cdist(x,c).argmin(dim=1)

def assign_roles(x: torch.Tensor, centroids: torch.Tensor):
    return torch.cdist(x,centroids).argmin(dim=1)
