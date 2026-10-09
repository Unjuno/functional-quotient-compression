import torch
from torch import nn
import torch.nn.functional as F

INPUT_DIM = 16
HIDDEN = 16
CLASSES = 4
NODES = 3
PATHS = tuple(range(1 << NODES))
METHODS = ("independent", "supernet", "scalar_path", "mirror_path")


class PathMLP(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.path = int(path)
        self.stem = nn.Linear(INPUT_DIM, HIDDEN)
        self.nodes = nn.ModuleDict({str(i): nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
                                    for i in range(NODES) if self.path & (1 << i)})
        self.readout = nn.Linear(HIDDEN, CLASSES)
        with torch.no_grad():
            for p in self.parameters():
                p.mul_(0.15)

    def forward(self, x):
        h = torch.tanh(self.stem(x))
        for i in range(NODES):
            if self.path & (1 << i):
                h = h + self.nodes[str(i)](h)
        return self.readout(h)


class IndependentChildren(nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.models = nn.ModuleDict({str(path): PathMLP(path) for path in PATHS})

    def forward_path(self, path, x):
        return self.models[str(path)](x)


class SharedDAG(nn.Module):
    def __init__(self, method, seed):
        super().__init__()
        assert method in ("supernet", "scalar_path", "mirror_path")
        self.method = method
        self.stem = nn.Linear(INPUT_DIM, HIDDEN)
        self.nodes = nn.ModuleList(nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
                                   for _ in range(NODES))
        self.readout = nn.Linear(HIDDEN, CLASSES)
        if method == "scalar_path":
            self.codes = nn.Parameter(torch.zeros(len(PATHS)))
        elif method == "mirror_path":
            self.codes = nn.Parameter(torch.zeros(len(PATHS)))
        torch.manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters():
                p.mul_(0.15)

    def hidden_path(self, path, x):
        h = torch.tanh(self.stem(x))
        for i, node in enumerate(self.nodes):
            if path & (1 << i):
                h = h + node(h)
        if self.method == "scalar_path":
            h = h * (1.0 + self.codes[path])
        elif self.method == "mirror_path":
            c, s = torch.cos(self.codes[path]), torch.sin(self.codes[path])
            even, odd = h[:, 0::2], h[:, 1::2]
            h = torch.stack((c * even - s * odd, s * even + c * odd), dim=-1).flatten(-2)
        return h

    def forward_path(self, path, x):
        return self.readout(self.hidden_path(path, x))
