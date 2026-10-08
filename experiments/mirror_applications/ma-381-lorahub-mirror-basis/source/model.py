import torch
from torch import nn

INPUT_DIM = 16
OUTPUT_DIM = 16
RANK = 2
SOURCE_TASKS = 8
TARGET_TASKS = 4
METHODS = ("independent", "hard_shared", "scalar_shared", "mirror_shared")


def givens(angle):
    c, s = torch.cos(angle), torch.sin(angle)
    r = torch.eye(INPUT_DIM, dtype=angle.dtype, device=angle.device)
    even = torch.arange(0, INPUT_DIM, 2, device=angle.device)
    odd = even + 1
    r[even, even] = c
    r[odd, odd] = c
    r[even, odd] = -s
    r[odd, even] = s
    return r


class CandidateBank(nn.Module):
    def __init__(self, method, seed):
        super().__init__()
        assert method in METHODS
        self.method = method
        g = torch.Generator().manual_seed(seed)
        if method == "independent":
            self.source_a = nn.Parameter(torch.randn(SOURCE_TASKS, INPUT_DIM, RANK, generator=g) * 0.08)
            self.source_b = nn.Parameter(torch.randn(SOURCE_TASKS, RANK, OUTPUT_DIM, generator=g) * 0.08)
        else:
            self.base_a = nn.Parameter(torch.randn(INPUT_DIM, RANK, generator=g) * 0.08)
            self.base_b = nn.Parameter(torch.randn(RANK, OUTPUT_DIM, generator=g) * 0.08)
            if method in ("scalar_shared", "mirror_shared"):
                self.codes = nn.Parameter(torch.zeros(SOURCE_TASKS))
        self.fusion = nn.Parameter(torch.zeros(TARGET_TASKS, SOURCE_TASKS), requires_grad=False)

    def source_matrices(self):
        if self.method == "independent":
            return torch.einsum("kir,kro->kio", self.source_a, self.source_b)
        base = self.base_a @ self.base_b
        if self.method == "hard_shared":
            return base.unsqueeze(0).expand(SOURCE_TASKS, -1, -1)
        if self.method == "scalar_shared":
            return (1.0 + self.codes[:, None, None]) * base[None, :, :]
        maps = []
        for angle in self.codes:
            r = givens(angle)
            maps.append(r @ base @ r.T)
        return torch.stack(maps)

    def source_outputs(self, x):
        return torch.einsum("bi,kio->bko", x, self.source_matrices())

    def target_outputs(self, x):
        outputs = self.source_outputs(x)
        return torch.einsum("tk,bko->bto", self.fusion, outputs)
