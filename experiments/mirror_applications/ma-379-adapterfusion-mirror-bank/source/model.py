import torch
from torch import nn

INPUT_DIM = 16
ADAPTER_DIM = 16
SOURCE_TASKS = 4
TARGET_TASKS = 4
METHODS = ("independent", "hard_tied", "scalar_shared", "mirror_shared")


def rotation(angle):
    c, s = torch.cos(angle), torch.sin(angle)
    eye = torch.eye(ADAPTER_DIM, dtype=angle.dtype, device=angle.device)
    matrix = eye.clone()
    even = torch.arange(0, ADAPTER_DIM, 2, device=angle.device)
    odd = even + 1
    matrix[even, even] = c
    matrix[odd, odd] = c
    matrix[even, odd] = -s
    matrix[odd, even] = s
    return matrix


class AdapterBank(nn.Module):
    def __init__(self, method, seed):
        super().__init__()
        assert method in METHODS
        self.method = method
        gen = torch.Generator().manual_seed(seed)
        if method == "independent":
            self.source_weights = nn.Parameter(torch.randn(SOURCE_TASKS, INPUT_DIM, ADAPTER_DIM, generator=gen) * 0.03)
        else:
            self.base = nn.Parameter(torch.randn(INPUT_DIM, ADAPTER_DIM, generator=gen) * 0.03)
            if method == "scalar_shared":
                self.source_codes = nn.Parameter(torch.zeros(SOURCE_TASKS))
            elif method == "mirror_shared":
                self.source_codes = nn.Parameter(torch.zeros(SOURCE_TASKS))
        self.fusion = nn.Parameter(torch.zeros(TARGET_TASKS, SOURCE_TASKS))

    def source_matrices(self):
        if self.method == "independent":
            return self.source_weights
        if self.method == "hard_tied":
            return self.base.unsqueeze(0).expand(SOURCE_TASKS, -1, -1)
        if self.method == "scalar_shared":
            return (1.0 + self.source_codes[:, None, None]) * self.base[None, :, :]
        matrices = []
        for angle in self.source_codes:
            r = rotation(angle)
            matrices.append(r @ self.base @ r.T)
        return torch.stack(matrices)

    def source_outputs(self, x):
        return torch.einsum("bd,kdf->bkf", x, self.source_matrices())

    def target_outputs(self, x):
        source = self.source_outputs(x)
        return torch.einsum("tk,bkf->btf", self.fusion, source)
