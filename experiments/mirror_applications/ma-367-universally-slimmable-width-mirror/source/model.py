import torch
from torch import nn
import torch.nn.functional as F

WIDTHS = (8, 16, 24, 32)
INPUT_DIM = 32
NUM_CLASSES = 4
MAX_WIDTH = 32
METHODS = ("independent", "usnet", "scalar_gate", "mirror_givens", "ia3_width")


def _init_linear(module, gen, scale):
    with torch.no_grad():
        module.weight.copy_(torch.randn(module.weight.shape, generator=gen) * scale)
        module.bias.zero_()


class WidthMLP(nn.Module):
    def __init__(self, width, seed):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(INPUT_DIM, width)
        self.fc2 = nn.Linear(width, NUM_CLASSES)
        gen = torch.Generator().manual_seed(seed)
        _init_linear(self.fc1, gen, 0.22)
        _init_linear(self.fc2, gen, 0.28)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class IndependentWidths(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.models = nn.ModuleDict({str(w): WidthMLP(w, seed + w * 17) for w in WIDTHS})

    def forward_width(self, width, x):
        return self.models[str(width)](x)


class SharedSlimmableMLP(nn.Module):
    def __init__(self, method, seed):
        super().__init__()
        assert method in ("usnet", "scalar_gate", "mirror_givens", "ia3_width")
        self.method = method
        self.w1 = nn.Parameter(torch.empty(INPUT_DIM, MAX_WIDTH))
        self.b1 = nn.Parameter(torch.zeros(MAX_WIDTH))
        self.w2 = nn.Parameter(torch.empty(MAX_WIDTH, NUM_CLASSES))
        self.b2 = nn.Parameter(torch.zeros(NUM_CLASSES))
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.w1.copy_(torch.randn(self.w1.shape, generator=gen) * 0.22)
            self.w2.copy_(torch.randn(self.w2.shape, generator=gen) * 0.28)
        if method == "scalar_gate":
            self.codes = nn.Parameter(torch.zeros(len(WIDTHS)))
        elif method == "mirror_givens":
            self.codes = nn.Parameter(torch.zeros(len(WIDTHS)))
        elif method == "ia3_width":
            self.codes = nn.Parameter(torch.zeros(len(WIDTHS), MAX_WIDTH))

    def hidden(self, x, width):
        idx = WIDTHS.index(width)
        h = F.relu(x @ self.w1[:, :width] + self.b1[:width])
        if self.method == "scalar_gate":
            h = h * (1.0 + self.codes[idx])
        elif self.method == "mirror_givens":
            c = torch.cos(self.codes[idx])
            s = torch.sin(self.codes[idx])
            even, odd = h[:, 0::2], h[:, 1::2]
            h = torch.stack((c * even - s * odd, s * even + c * odd), dim=-1).flatten(-2)
        elif self.method == "ia3_width":
            h = h * (1.0 + self.codes[idx, :width])
        return h

    def forward_width(self, width, x):
        h = self.hidden(x, width)
        return h @ self.w2[:width, :] + self.b2
