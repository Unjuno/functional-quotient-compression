"""Small sequential linear expert models for MA-010."""
import torch
from torch import nn


METHODS = ("independent", "tied", "rank2", "scalar", "mirror")
DIM = 8
ROLES = 4


class SequentialExperts(nn.Module):
    def __init__(self, method: str, init_seed: int):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        torch.manual_seed(init_seed)
        if method == "independent":
            self.weight = nn.Parameter(torch.randn(ROLES, DIM, DIM) * 0.12)
        else:
            self.weight = nn.Parameter(torch.randn(DIM, DIM) * 0.12)
            if method == "rank2":
                self.left = nn.Parameter(torch.randn(ROLES, DIM, 2) * 0.03)
                self.right = nn.Parameter(torch.randn(ROLES, 2, DIM) * 0.03)
            elif method == "scalar":
                self.gate = nn.Parameter(torch.ones(ROLES))
            elif method == "mirror":
                self.input_angle = nn.Parameter(torch.zeros(ROLES))
                self.output_angle = nn.Parameter(torch.zeros(ROLES))

    def matrices(self):
        if self.method == "independent":
            return self.weight
        base = self.weight.unsqueeze(0).expand(ROLES, -1, -1)
        if self.method == "tied":
            return base
        if self.method == "rank2":
            return base + self.left @ self.right
        if self.method == "scalar":
            return base * self.gate[:, None, None]
        if self.method == "mirror":
            # R_out(theta) W R_in(theta)^T, with a single 2D Givens plane.
            wi = _rotation(self.input_angle, transpose=True)
            wo = _rotation(self.output_angle, transpose=False)
            return wo @ base @ wi
        raise AssertionError(self.method)

    def forward(self, x, roles):
        matrices = self.matrices()
        h = x
        # The tuple order is functional: role a is applied before role b.
        for step in range(roles.shape[1]):
            w = matrices[roles[:, step]]
            h = torch.bmm(w, h.unsqueeze(-1)).squeeze(-1)
        return h


def _rotation(angles, transpose=False):
    c, s = torch.cos(angles), torch.sin(angles)
    if transpose:
        s = -s
    zero = torch.zeros_like(c)
    one = torch.ones_like(c)
    # Batched 8x8 identity with the first 2x2 block replaced by a rotation.
    eye = torch.eye(DIM, dtype=angles.dtype, device=angles.device).expand(len(angles), -1, -1).clone()
    eye[:, 0, 0] = c
    eye[:, 0, 1] = -s
    eye[:, 1, 0] = s
    eye[:, 1, 1] = c
    return eye

