import torch
from torch import nn

D, C, R = 64, 8, 4


def rotate(q, angles):
    out = q.clone()
    a, b = angles[:, 0:1], angles[:, 1:2]
    ca, sa = torch.cos(a), torch.sin(a)
    x, y = out[:, 0:1].clone(), out[:, 1:2].clone()
    out[:, 0:1], out[:, 1:2] = ca*x-sa*y, sa*x+ca*y
    cb, sb = torch.cos(b), torch.sin(b)
    x, y = out[:, 2:3].clone(), out[:, 3:4].clone()
    out[:, 2:3], out[:, 3:4] = cb*x-sb*y, sb*x+cb*y
    return out


class InterventionModel(nn.Module):
    def __init__(self, method, basis):
        super().__init__()
        self.method = method
        if method in {'mirror_givens', 'linear_loreft', 'mlp_loreft'}:
            self.register_buffer('basis', basis.clone())
        if method == 'mirror_givens':
            self.proj = nn.Linear(D, R, bias=False)
            self.angle = nn.Linear(C, 2, bias=False)
        elif method == 'linear_loreft':
            self.code = nn.Linear(D+C, R)
        elif method == 'mlp_loreft':
            self.code = nn.Sequential(nn.Linear(D+C, 32), nn.Tanh(), nn.Linear(32, R))
        elif method == 'film':
            self.affine = nn.Sequential(nn.Linear(C, 32), nn.Tanh(), nn.Linear(32, 2*D))
            nn.init.zeros_(self.affine[-1].weight)
            nn.init.zeros_(self.affine[-1].bias)
        elif method == 'shared':
            pass
        else:
            raise ValueError(method)

    def forward(self, h, u):
        if self.method == 'shared':
            return torch.zeros_like(h)
        if self.method == 'mirror_givens':
            q = self.proj(h)
            angles = self.angle(u)
            return rotate(q, angles) @ self.basis.T
        if self.method == 'linear_loreft':
            return self.code(torch.cat((h,u), dim=-1)) @ self.basis.T
        if self.method == 'mlp_loreft':
            return self.code(torch.cat((h,u), dim=-1)) @ self.basis.T
        if self.method == 'film':
            gamma, beta = self.affine(u).chunk(2, dim=-1)
            return (1.0 + gamma) * h + beta - h
        raise ValueError(self.method)


METHODS = ['shared', 'linear_loreft', 'mirror_givens', 'mlp_loreft', 'film']
MACS = {'shared': 0, 'linear_loreft': 544, 'mirror_givens': 536, 'mlp_loreft': 2688, 'film': 4416}
