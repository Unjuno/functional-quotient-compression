# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Evaluation helpers reused from MN001; unchanged semantics."""
import json
import platform
from pathlib import Path
import torch
from torch.nn import functional as F

def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

@torch.no_grad()
def evaluate(model, data, override=None, batch_size=128):
    x, y = data
    nll, correct, count = 0., 0, 0
    seq_errors, route_sums, route_sums_all, route_entropy, route_argmax, route_count, route_count_all = [], [], [], [], [], 0, 0
    for i in range(0, len(x), batch_size):
        xb, yb = x[i:i+batch_size], y[i:i+batch_size]
        logits, _, routes = model(xb, override=override)
        mask = yb != -100
        loss = F.cross_entropy(logits[mask].double(), yb[mask], reduction='none')
        nll += loss.sum().item()
        correct += (logits.argmax(-1)[mask] == yb[mask]).sum().item()
        count += int(mask.sum())
        seq_errors.extend(loss.reshape(len(xb), -1).mean(-1).tolist())
        if routes:
            if not route_sums:
                route_sums = [torch.zeros(p.shape[-1], dtype=torch.float64) for p in routes]
                route_sums_all = [torch.zeros(p.shape[-1], dtype=torch.float64) for p in routes]
                route_entropy = [0. for p in routes]
                route_argmax = [torch.zeros(p.shape[-1], dtype=torch.long) for p in routes]
            for l, p in enumerate(routes):
                pp = p[mask].double()
                route_sums[l] += pp.sum(0)
                route_sums_all[l] += p.double().sum((0,1))
                route_entropy[l] += (-(pp * pp.clamp_min(1e-30).log()).sum(-1)).sum().item()
                route_argmax[l] += torch.bincount(pp.argmax(-1), minlength=pp.shape[-1])
            route_count += int(mask.sum())
            route_count_all += xb.numel()
    result = dict(nll=nll/count, accuracy=correct/count, answer_tokens=count,
                  sequences=len(x), per_sequence_nll=seq_errors)
    if routes:
        result['routing'] = [dict(mean_prob=(s/route_count).tolist(),
              mean_entropy=ent/route_count, argmax_counts=c.tolist(),
              mean_prob_all_tokens=(sa/route_count_all).tolist())
              for s, sa, ent, c in zip(route_sums, route_sums_all, route_entropy, route_argmax)]
    return result

def environment():
    cpu = Path('/proc/cpuinfo').read_text()
    return dict(python=platform.python_version(), torch=torch.__version__,
                platform=platform.platform(), cpu=next(l.split(':',1)[1].strip() for l in cpu.splitlines() if l.startswith('model name')),
                observed_mhz=[float(l.split(':')[1]) for l in cpu.splitlines() if l.startswith('cpu MHz')],
                clock_locked=False, cpu_threads=torch.get_num_threads(), cuda=torch.cuda.is_available(),
                dtype='float32 forward/training; float64 reported cross-entropy')
