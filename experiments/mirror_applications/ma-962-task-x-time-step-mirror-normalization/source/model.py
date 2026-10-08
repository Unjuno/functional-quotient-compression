"""Small task-conditioned SNN with native TEBN, cyclic-phase Mirror and rank-1 control."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

N_TASKS, TIMESTEPS, HIDDEN, INPUTS, CLASSES = 4, 8, 128, 784, 10
ENVELOPE = torch.tensor([0.25, 0.40, 0.75, 1.20, 1.50, 1.10, 0.55, 0.25], dtype=torch.float32)
PHASES = torch.tensor([0.0, 2.0, 4.0, 6.0], dtype=torch.float32)


class FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1.0 + x.abs()).square()


def encode_rate_spikes(images: torch.Tensor, task_ids: torch.Tensor, seed: int) -> torch.Tensor:
    """Sample matched Bernoulli event trains. images are [B,784] floats in [0,1]."""
    gen = torch.Generator(device='cpu').manual_seed(int(seed))
    env = ENVELOPE[(torch.arange(TIMESTEPS)[None, :] - PHASES[task_ids].long()[:, None]) % TIMESTEPS]
    p = (0.65 * images[:, None, :] * env[:, :, None]).clamp(max=0.9)
    return (torch.rand(p.shape, generator=gen) < p).to(images.dtype)


class SharedSNN(nn.Module):
    def __init__(self, condition: str):
        super().__init__()
        if condition not in {'shared_no_task_gain', 'native_task_time_tebn', 'mirror_cyclic_phase', 'rank1_task_scalar_gate_on_shared_tebn_profile'}:
            raise ValueError(condition)
        self.condition = condition
        self.input = nn.Linear(INPUTS, HIDDEN)
        self.readout = nn.Linear(HIDDEN, CLASSES)
        if condition == 'native_task_time_tebn':
            self.log_gain = nn.Parameter(torch.zeros(N_TASKS, TIMESTEPS, HIDDEN))
        elif condition == 'mirror_cyclic_phase':
            self.log_gain_profile = nn.Parameter(torch.zeros(TIMESTEPS, HIDDEN))
            self.phase = nn.Parameter(PHASES.clone())
        elif condition == 'rank1_task_scalar_gate_on_shared_tebn_profile':
            self.log_gain_profile = nn.Parameter(torch.zeros(TIMESTEPS, HIDDEN))
            self.task_code = nn.Parameter(torch.zeros(N_TASKS, 1))
        else:
            self.log_gain_profile = nn.Parameter(torch.zeros(TIMESTEPS, HIDDEN))

    def gain(self, task_ids: torch.Tensor) -> torch.Tensor:
        if self.condition == 'native_task_time_tebn':
            log_gain = self.log_gain.index_select(0, task_ids)
        elif self.condition == 'mirror_cyclic_phase':
            t = torch.arange(TIMESTEPS, device=task_ids.device, dtype=self.phase.dtype)[None, :]
            pos = torch.remainder(t - self.phase[task_ids, None], TIMESTEPS)
            lo = torch.floor(pos).long()
            frac = (pos - lo.to(pos.dtype)).unsqueeze(-1)
            hi = (lo + 1) % TIMESTEPS
            profile = torch.exp(self.log_gain_profile.clamp(-2.0, 2.0))
            left = profile[lo]
            right = profile[hi]
            return left + frac * (right - left)
        elif self.condition == 'rank1_task_scalar_gate_on_shared_tebn_profile':
            log_gain = self.log_gain_profile[None] + self.task_code[task_ids, 0, None, None]
        else:
            log_gain = self.log_gain_profile[None].expand(task_ids.shape[0], -1, -1)
        return torch.exp(log_gain.clamp(-2.0, 2.0))

    def forward(self, spikes: torch.Tensor, task_ids: torch.Tensor):
        batch = spikes.shape[0]
        gain = self.gain(task_ids)
        membrane = spikes.new_zeros((batch, HIDDEN))
        logits = spikes.new_zeros((batch, CLASSES))
        spike_count = spikes.new_zeros(())
        for t in range(TIMESTEPS):
            current = self.input(spikes[:, t]) * gain[:, t]
            membrane = 0.5 * membrane + current
            fired = FastSigmoidSpike.apply(membrane - 1.0)
            membrane = membrane - fired.detach()
            logits = logits + self.readout(fired)
            spike_count = spike_count + fired.detach().sum()
        return logits / TIMESTEPS, spike_count / batch


class IndependentTaskSNN(nn.Module):
    def __init__(self, seed: int | None = None):
        super().__init__()
        self.models = nn.ModuleList()
        for task in range(N_TASKS):
            if seed is not None:
                torch.manual_seed(seed + task)
            self.models.append(SharedSNN('shared_no_task_gain'))

    def forward_task(self, spikes: torch.Tensor, task: int):
        tasks = torch.full((spikes.shape[0],), task, dtype=torch.long, device=spikes.device)
        return self.models[task](spikes, tasks)


def inference_macs_per_example() -> int:
    return TIMESTEPS * (INPUTS * HIDDEN + HIDDEN * CLASSES)
