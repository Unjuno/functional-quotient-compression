"""MNS-001 L0: tiny causal shared projections with parallel Mirror view codes.

This is an analytical mechanism probe, NOT a trained sequence language model.
"""
import io
import math
import torch
from torch import nn
from torch.nn import functional as F


def rotate_first_plane(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Givens rotation in the first 2D feature plane; remaining channels unchanged."""
    c, s = torch.cos(angle), torch.sin(angle)
    a, b = x[..., 0], x[..., 1]
    return torch.cat(((c * a - s * b)[..., None], (s * a + c * b)[..., None], x[..., 2:]), -1)


def conjugated_relu(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    return rotate_first_plane(F.relu(rotate_first_plane(x, angle)), -angle)


class SharedMirrorCausal(nn.Module):
    """One physical QKV projection triplet, K logical causal attention views.

    Each view uses distinct q/k rotations and a nonlinear v transform.
    Scores are causal; value/output coordinate conversion uses one shared
    angle per view with no extra learned projection per view.
    """

    def __init__(self, d: int = 8, views: int = 4):
        super().__init__()
        if d < 2 or views < 1:
            raise ValueError("d>=2, views>=1 required")
        self.d, self.views = d, views
        self.wq = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.wk = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.wv = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.q_angle = nn.Parameter(torch.linspace(-0.25, 0.25, views))
        self.k_angle = nn.Parameter(torch.linspace(0.36, -0.36, views))
        self.v_angle = nn.Parameter(torch.linspace(-0.48, 0.48, views))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.d:
            raise ValueError("expected [batch,sequence,d]")
        q0, k0, v0 = x @ self.wq, x @ self.wk, x @ self.wv
        length = x.shape[1]
        upper = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        outputs = []
        for j in range(self.views):
            q = rotate_first_plane(q0, self.q_angle[j])
            k = rotate_first_plane(k0, self.k_angle[j])
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.d)
            att = torch.softmax(scores.masked_fill(upper, -torch.inf), dim=-1)
            # Nonlinearity is inside the V-view sandwich, avoids trivial cancellation.
            v = conjugated_relu(v0, self.v_angle[j])
            outputs.append(att @ v)
        return torch.stack(outputs, dim=1)  # [B,view,T,D]


def inference_payload_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save({'cfg': {'d': model.d, 'views': model.views},
                'state': model.state_dict()}, buf)
    return len(buf.getvalue())
