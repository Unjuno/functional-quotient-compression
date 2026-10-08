import torch
from torch import nn
import torch.nn.functional as F

WIDTHS = (8, 16, 24, 32)
DEPTHS = (1, 2, 3, 4)
HELD_OUT = ((8, 3), (16, 4), (24, 1), (32, 2))
INPUT_DIM = 32
NUM_CLASSES = 4
METHODS = ("independent", "supernet", "factor_scalar", "factor_mirror")


def _copy_random(parameter, generator, scale):
    with torch.no_grad():
        parameter.copy_(torch.randn(parameter.shape, generator=generator) * scale)


class ConfigMLP(nn.Module):
    def __init__(self, width, depth, seed):
        super().__init__()
        self.width, self.depth = width, depth
        self.first = nn.Linear(INPUT_DIM, width)
        self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 1))
        self.readout = nn.Linear(width, NUM_CLASSES)
        gen = torch.Generator().manual_seed(seed)
        _copy_random(self.first.weight, gen, 0.15)
        _copy_random(self.readout.weight, gen, 0.15)
        for layer in self.hidden:
            _copy_random(layer.weight, gen, 0.15)
        with torch.no_grad():
            for module in (self.first, *self.hidden, self.readout):
                module.bias.zero_()

    def forward(self, x):
        h = F.relu(self.first(x))
        for layer in self.hidden:
            h = F.relu(layer(h))
        return self.readout(h)


class IndependentBank(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.models = nn.ModuleDict({
            f"w{w}_d{d}": ConfigMLP(w, d, seed + w * 97 + d * 13)
            for w in WIDTHS for d in DEPTHS
        })

    def forward_config(self, width, depth, x):
        return self.models[f"w{width}_d{depth}"](x)


class ElasticSupernet(nn.Module):
    def __init__(self, method, seed):
        super().__init__()
        assert method in ("supernet", "factor_scalar", "factor_mirror")
        self.method = method
        self.w_in = nn.Parameter(torch.empty(INPUT_DIM, max(WIDTHS)))
        self.b_in = nn.Parameter(torch.zeros(max(WIDTHS)))
        self.w_hidden = nn.ParameterList(nn.Parameter(torch.empty(max(WIDTHS), max(WIDTHS))) for _ in range(max(DEPTHS) - 1))
        self.b_hidden = nn.ParameterList(nn.Parameter(torch.zeros(max(WIDTHS))) for _ in range(max(DEPTHS) - 1))
        self.w_out = nn.Parameter(torch.empty(max(WIDTHS), NUM_CLASSES))
        self.b_out = nn.Parameter(torch.zeros(NUM_CLASSES))
        gen = torch.Generator().manual_seed(seed)
        _copy_random(self.w_in, gen, 0.15)
        _copy_random(self.w_out, gen, 0.15)
        for weight in self.w_hidden:
            _copy_random(weight, gen, 0.15)
        if method != "supernet":
            self.width_codes = nn.Parameter(torch.zeros(len(WIDTHS)))
            self.depth_codes = nn.Parameter(torch.zeros(len(DEPTHS)))

    def forward_config(self, width, depth, x):
        h = F.relu(x @ self.w_in[:, :width] + self.b_in[:width])
        for layer in range(depth - 1):
            h = F.relu(h @ self.w_hidden[layer][:width, :width] + self.b_hidden[layer][:width])
        if self.method == "factor_scalar":
            amount = self.width_codes[WIDTHS.index(width)] + self.depth_codes[DEPTHS.index(depth)]
            h = h * (1.0 + amount)
        elif self.method == "factor_mirror":
            amount = self.width_codes[WIDTHS.index(width)] + self.depth_codes[DEPTHS.index(depth)]
            c, s = torch.cos(amount), torch.sin(amount)
            even, odd = h[:, 0::2], h[:, 1::2]
            h = torch.stack((c * even - s * odd, s * even + c * odd), dim=-1).flatten(-2)
        return h @ self.w_out[:width, :] + self.b_out
