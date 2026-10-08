"""Product-factor expert bank used by MA-011."""
import torch
from torch import nn

DIM, ROLES = 8, 4
METHODS = ("independent", "tied", "rank2", "scalar", "mirror")


def rotate(angle, transpose=False):
    c, s = torch.cos(angle), torch.sin(angle)
    if transpose:
        s = -s
    out = torch.eye(DIM, device=angle.device, dtype=angle.dtype).expand(*angle.shape, DIM, DIM).clone()
    out[..., 0, 0] = c; out[..., 0, 1] = -s
    out[..., 1, 0] = s; out[..., 1, 1] = c
    return out


class ProductExperts(nn.Module):
    def __init__(self, method, seed):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        torch.manual_seed(seed)
        if method == "independent":
            self.weight = nn.Parameter(torch.randn(ROLES, 2, DIM, DIM) * 0.10)
        else:
            self.weight = nn.Parameter(torch.randn(2, DIM, DIM) * 0.10)
            if method == "rank2":
                self.left = nn.Parameter(torch.randn(2, ROLES, DIM, 2) * 0.025)
                self.right = nn.Parameter(torch.randn(2, ROLES, 2, DIM) * 0.025)
            elif method == "scalar":
                self.gate = nn.Parameter(torch.ones(2, ROLES))
            elif method == "mirror":
                self.input_angle = nn.Parameter(torch.zeros(2, 2))
                self.output_angle = nn.Parameter(torch.zeros(2, 2))

    def factors(self):
        if self.method == "independent":
            return self.weight.permute(1, 0, 2, 3)
        base = self.weight[:, None].expand(-1, ROLES, -1, -1)
        if self.method == "tied":
            return base
        if self.method == "rank2":
            return base + self.left @ self.right
        if self.method == "scalar":
            return base * self.gate[:, :, None, None]
        if self.method == "mirror":
            codes = (torch.tensor([0, 0, 1, 1], device=base.device), torch.tensor([0, 1, 0, 1], device=base.device))
            return torch.stack([
                rotate(self.output_angle[i, codes[i]]) @ base[i] @ rotate(self.input_angle[i, codes[i]], transpose=True)
                for i in range(2)
            ])
        raise AssertionError(self.method)

    def forward(self, x, role):
        ws = self.factors()
        ia, ib = torch.div(role, 2, rounding_mode="floor"), role.remainder(2)
        a = torch.tanh(torch.bmm(ws[0, ia], x.unsqueeze(-1)).squeeze(-1))
        b = torch.tanh(torch.bmm(ws[1, ib], x.unsqueeze(-1)).squeeze(-1))
        return a * b

