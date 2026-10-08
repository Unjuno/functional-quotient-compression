"""MA-446 frozen CPU mechanism screen. No third_party baseline files are changed."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cpu")
DTYPE = torch.float32
INPUT_DIM, OUTPUT_DIM, RANK = 4, 3, 3
STEPS, SUPPORT_N, QUERY_N = 4, 16, 128
META_TASKS, OUTER_UPDATES, HIDDEN = 16, 256, 8
SIGMAS = torch.tensor([0.2, 0.5, 1.0], dtype=DTYPE)
LRS = (0.03, 0.1, 0.3, 1.0)


def generator(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def make_world(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return shared W0 and orthonormal 3-matrix basis using one fixed seed."""
    g = generator(seed)
    w0 = 0.1 * torch.randn(OUTPUT_DIM, INPUT_DIM, generator=g, dtype=DTYPE)
    raw = torch.randn(INPUT_DIM * OUTPUT_DIM, RANK, generator=g, dtype=DTYPE)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    basis = q.T.reshape(RANK, OUTPUT_DIM, INPUT_DIM).contiguous()
    return w0, basis


def sample_targets(n: int, w0: torch.Tensor, basis: torch.Tensor,
                   g: torch.Generator) -> torch.Tensor:
    z = torch.randn(n, RANK, generator=g, dtype=DTYPE) * SIGMAS
    return w0.unsqueeze(0) + torch.einsum("bk,koi->boi", z, basis)


def make_episode(n: int, w0: torch.Tensor, basis: torch.Tensor,
                 g: torch.Generator) -> dict[str, torch.Tensor]:
    targets = sample_targets(n, w0, basis, g)
    sx = torch.randn(n, STEPS, SUPPORT_N, INPUT_DIM, generator=g, dtype=DTYPE)
    sy = torch.einsum("btni,boi->btno", sx, targets)
    sy += 0.05 * torch.randn(sy.shape, generator=g, dtype=DTYPE)
    qx = torch.randn(n, QUERY_N, INPUT_DIM, generator=g, dtype=DTYPE)
    qy = torch.einsum("bni,boi->bno", qx, targets)
    qy += 0.05 * torch.randn(qy.shape, generator=g, dtype=DTYPE)
    return {"targets": targets, "support_x": sx, "support_y": sy,
            "query_x": qx, "query_y": qy}


def decode_m(w0: torch.Tensor, basis: torch.Tensor,
             m: torch.Tensor) -> torch.Tensor:
    return w0.unsqueeze(0) + torch.einsum("bk,koi->boi", m, basis)


class CoordinateLSTM(nn.Module):
    """One recurrent update rule shared over every scalar being optimized."""
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.cell = nn.LSTMCell(2, hidden)
        self.delta = nn.Linear(hidden, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def initial_state(self, shape: tuple[int, int],
                      ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, n = shape
        h = ref.new_zeros(b, n, self.cell.hidden_size)
        c = ref.new_zeros(b, n, self.cell.hidden_size)
        return h, c

    def forward(self, grad: torch.Tensor,
                state: tuple[torch.Tensor, torch.Tensor]
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, c = state
        g = grad.clamp(-1.0e6, 1.0e6)
        feat = torch.stack((g.sign() * torch.log1p(g.abs()),
                            torch.log1p(g.abs())), dim=-1)
        b, n, _ = feat.shape
        h1, c1 = self.cell(feat.reshape(b*n, 2),
                           (h.reshape(b*n, -1), c.reshape(b*n, -1)))
        delta = self.delta(h1).reshape(b, n)
        return delta, (h1.reshape(b, n, -1), c1.reshape(b, n, -1))


def init_lstm(seed: int) -> CoordinateLSTM:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return CoordinateLSTM().to(device=DEVICE, dtype=DTYPE)


def _support_loss_m(m: torch.Tensor, w0: torch.Tensor,
                    basis: torch.Tensor, x: torch.Tensor,
                    y: torch.Tensor) -> torch.Tensor:
    w = decode_m(w0, basis, m)
    pred = torch.einsum("bni,boi->bno", x, w)
    return (pred - y).square().mean()


def _support_loss_w(w: torch.Tensor, x: torch.Tensor,
                    y: torch.Tensor) -> torch.Tensor:
    pred = torch.einsum("bni,boi->bno", x, w)
    return (pred - y).square().mean()


def train_lstm(task_seed: int, w0: torch.Tensor, basis: torch.Tensor,
               optimize_m: bool, init_seed: int) -> tuple[CoordinateLSTM, float, list[float]]:
    """Meta-train a recurrent optimizer by differentiating through 4 updates."""
    model = init_lstm(init_seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = generator(task_seed + 1_000_000)
    losses: list[float] = []
    start = time.perf_counter()
    for _ in range(OUTER_UPDATES):
        targets = sample_targets(META_TASKS, w0, basis, g)
        sx = torch.randn(META_TASKS, STEPS, SUPPORT_N, INPUT_DIM,
                         generator=g, dtype=DTYPE)
        sy = torch.einsum("btni,boi->btno", sx, targets)
        sy = sy + 0.05 * torch.randn(sy.shape, generator=g, dtype=DTYPE)
        qx = torch.randn(META_TASKS, QUERY_N, INPUT_DIM, generator=g, dtype=DTYPE)
        qy = torch.einsum("bni,boi->bno", qx, targets)
        qy = qy + 0.05 * torch.randn(qy.shape, generator=g, dtype=DTYPE)
        if optimize_m:
            p = torch.zeros(META_TASKS, RANK, dtype=DTYPE, requires_grad=True)
        else:
            p = w0.unsqueeze(0).expand(META_TASKS, -1, -1).clone().requires_grad_(True)
        flat = p.reshape(META_TASKS, -1)
        state = model.initial_state((META_TASKS, flat.shape[1]), flat)
        for step in range(STEPS):
            loss = (_support_loss_m(p, w0, basis, sx[:, step], sy[:, step])
                    if optimize_m else _support_loss_w(p, sx[:, step], sy[:, step]))
            grad = torch.autograd.grad(loss, p, create_graph=True)[0]
            flat_grad = grad.reshape(META_TASKS, -1)
            delta, state = model(flat_grad, state)
            p = (p.reshape(META_TASKS, -1) + delta).reshape_as(p)
        qloss = (_support_loss_m(p, w0, basis, qx, qy)
                 if optimize_m else _support_loss_w(p, qx, qy))
        opt.zero_grad(set_to_none=True)
        qloss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(float(qloss.detach()))
    return model, time.perf_counter() - start, losses


def train_meta_sgd(task_seed: int, w0: torch.Tensor,
                   basis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, list[float]]:
    """Meta-learn code initialization and diagonal update scales (PA151)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(task_seed + 17)
        m0 = nn.Parameter(torch.zeros(RANK, dtype=DTYPE))
        raw_alpha = nn.Parameter(torch.full((RANK,), -1.5, dtype=DTYPE))
    opt = torch.optim.Adam([m0, raw_alpha], lr=1e-3)
    g = generator(task_seed + 1_000_000)
    losses: list[float] = []
    start = time.perf_counter()
    for _ in range(OUTER_UPDATES):
        targets = sample_targets(META_TASKS, w0, basis, g)
        sx = torch.randn(META_TASKS, STEPS, SUPPORT_N, INPUT_DIM,
                         generator=g, dtype=DTYPE)
        sy = torch.einsum("btni,boi->btno", sx, targets)
        sy = sy + 0.05 * torch.randn(sy.shape, generator=g, dtype=DTYPE)
        qx = torch.randn(META_TASKS, QUERY_N, INPUT_DIM, generator=g, dtype=DTYPE)
        qy = torch.einsum("bni,boi->bno", qx, targets)
        qy = qy + 0.05 * torch.randn(qy.shape, generator=g, dtype=DTYPE)
        m = m0.unsqueeze(0).expand(META_TASKS, -1)
        alpha = F.softplus(raw_alpha).clamp_max(2.0)
        for step in range(STEPS):
            loss = _support_loss_m(m, w0, basis, sx[:, step], sy[:, step])
            grad = torch.autograd.grad(loss, m, create_graph=True)[0]
            m = m - alpha.unsqueeze(0) * grad
        qloss = _support_loss_m(m, w0, basis, qx, qy)
        opt.zero_grad(set_to_none=True)
        qloss.backward()
        opt.step()
        losses.append(float(qloss.detach()))
    return m0.detach().clone(), F.softplus(raw_alpha.detach()).clamp_max(2.0), time.perf_counter()-start, losses


def query_mse_m(m: torch.Tensor, w0: torch.Tensor, basis: torch.Tensor,
                qx: torch.Tensor, qy: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        w = decode_m(w0, basis, m)
        return (torch.einsum("bni,boi->bno", qx, w)-qy).square().mean(dim=(1,2))


def query_mse_w(w: torch.Tensor, qx: torch.Tensor,
                qy: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return (torch.einsum("bni,boi->bno", qx, w)-qy).square().mean(dim=(1,2))


def adapt_sgd_or_adam(episode: dict[str, torch.Tensor], w0: torch.Tensor,
                      basis: torch.Tensor, optimize_m: bool, method: str,
                      lr: float) -> dict[str, Any]:
    n = episode['support_x'].shape[0]
    p = (torch.zeros(n, RANK, dtype=DTYPE) if optimize_m else
         w0.unsqueeze(0).expand(n, -1, -1).clone())
    m1, m2 = torch.zeros_like(p), torch.zeros_like(p)
    q0=(query_mse_m(p,w0,basis,episode['query_x'],episode['query_y']) if optimize_m else query_mse_w(p,episode['query_x'],episode['query_y']))
    task_curves=[q0.tolist()]; curve=[float(q0.mean())]
    start = time.perf_counter()
    for step in range(STEPS):
        p = p.detach().requires_grad_(True)
        loss = (_support_loss_m(p, w0, basis, episode['support_x'][:,step], episode['support_y'][:,step])
                if optimize_m else _support_loss_w(p, episode['support_x'][:,step], episode['support_y'][:,step]))
        grad = torch.autograd.grad(loss, p)[0]
        with torch.no_grad():
            if method == 'sgd':
                p = p - lr * grad
            elif method == 'adam':
                m1.mul_(0.9).add_(grad, alpha=0.1)
                m2.mul_(0.999).addcmul_(grad, grad, value=0.001)
                mhat = m1 / (1 - 0.9**(step+1))
                vhat = m2 / (1 - 0.999**(step+1))
                p = p - lr * mhat / (vhat.sqrt() + 1e-8)
            else:
                raise ValueError(method)
        q=(query_mse_m(p,w0,basis,episode['query_x'],episode['query_y']) if optimize_m else query_mse_w(p,episode['query_x'],episode['query_y']))
        task_curves.append(q.tolist()); curve.append(float(q.mean()))
    elapsed = time.perf_counter()-start
    result={'curve':curve,'task_curves':task_curves,'params':p.detach(),'wall_seconds_per_task':elapsed/n}
    if method == 'adam': result.update({'m1':m1,'m2':m2})
    return result


def adapt_meta_sgd(episode: dict[str, torch.Tensor], w0: torch.Tensor,
                   basis: torch.Tensor, init: torch.Tensor,
                   alpha: torch.Tensor) -> dict[str, Any]:
    n=episode['support_x'].shape[0]; p=init.unsqueeze(0).expand(n,-1).clone()
    q0=query_mse_m(p,w0,basis,episode['query_x'],episode['query_y']); task_curves=[q0.tolist()]; curve=[float(q0.mean())]
    start=time.perf_counter()
    for step in range(STEPS):
        p=p.detach().requires_grad_(True)
        loss=_support_loss_m(p,w0,basis,episode['support_x'][:,step],episode['support_y'][:,step])
        grad=torch.autograd.grad(loss,p)[0]
        with torch.no_grad(): p=p-alpha.unsqueeze(0)*grad
        q=query_mse_m(p,w0,basis,episode['query_x'],episode['query_y']); task_curves.append(q.tolist()); curve.append(float(q.mean()))
    return {'curve':curve,'task_curves':task_curves,'params':p.detach(),'wall_seconds_per_task':(time.perf_counter()-start)/n}


def adapt_lstm(model: CoordinateLSTM, episode: dict[str, torch.Tensor],
               w0: torch.Tensor, basis: torch.Tensor,
               optimize_m: bool) -> dict[str, Any]:
    n=episode['support_x'].shape[0]
    p=(torch.zeros(n,RANK,dtype=DTYPE) if optimize_m else
       w0.unsqueeze(0).expand(n,-1,-1).clone())
    flat=p.reshape(n,-1)
    state=model.initial_state((n,flat.shape[1]),flat)
    q0=(query_mse_m(p,w0,basis,episode['query_x'],episode['query_y']) if optimize_m else query_mse_w(p,episode['query_x'],episode['query_y']))
    task_curves=[q0.tolist()]; curve=[float(q0.mean())]
    start=time.perf_counter()
    for step in range(STEPS):
        p=p.detach().requires_grad_(True)
        loss=(_support_loss_m(p,w0,basis,episode['support_x'][:,step],episode['support_y'][:,step]) if optimize_m else _support_loss_w(p,episode['support_x'][:,step],episode['support_y'][:,step]))
        grad=torch.autograd.grad(loss,p)[0].reshape(n,-1)
        delta,state=model(grad,state)
        with torch.no_grad(): p=(p.reshape(n,-1)+delta).reshape_as(p)
        q=(query_mse_m(p,w0,basis,episode['query_x'],episode['query_y']) if optimize_m else query_mse_w(p,episode['query_x'],episode['query_y']))
        task_curves.append(q.tolist()); curve.append(float(q.mean()))
    return {'curve':curve,'task_curves':task_curves,'params':p.detach(),'state':tuple(x.detach() for x in state),'wall_seconds_per_task':(time.perf_counter()-start)/n}


def compute_proxy(method: str, n_parameters: int, mirror: bool) -> int:
    """Multiply-add proxy per task for four 16-example updates."""
    per_update = 2 * SUPPORT_N * INPUT_DIM * OUTPUT_DIM
    if mirror:
        per_update += 2 * RANK * INPUT_DIM * OUTPUT_DIM
    if method in ('lstm_mirror','lstm_full'):
        per_update += n_parameters * (4*HIDDEN*(2+HIDDEN) + HIDDEN)
    elif method == 'adam':
        per_update += 10*n_parameters
    else:
        per_update += n_parameters
    return STEPS*per_update


def tensor_bytes(obj: Any) -> bytes:
    stream=io.BytesIO(); torch.save(obj,stream); return stream.getvalue()


def save_library(path: Path, payload: dict[str, Any],
                 eval_episode: dict[str, torch.Tensor], w0: torch.Tensor,
                 basis: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    blob=tensor_bytes(payload); path.write_bytes(blob)
    restored=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
    if restored['method'] in ('full_adam','lstm_full'):
        weights=restored['states']['params']
        pred=torch.einsum('bni,boi->bno',eval_episode['query_x'][:8],weights)
    elif restored['method']=='hard_shared':
        pred=torch.einsum('bni,oi->bno',eval_episode['query_x'][:8],restored['shared']['w0'])
    else:
        m=restored['states']['params']
        w=decode_m(restored['shared']['w0'],restored['shared']['basis'],m)
        pred=torch.einsum('bni,boi->bno',eval_episode['query_x'][:8],w)
    diff=float((pred-expected).abs().max())
    if diff != 0.0: raise AssertionError(f'payload replay difference {diff}')
    return {'path':str(path.relative_to(ROOT)),'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'roundtrip_max_prediction_difference':diff}



def make_inference_payload(method: str, world_seed: int, w0: torch.Tensor,
                          basis: torch.Tensor, result: dict[str, Any]) -> dict[str, Any]:
    """Deployment-only state; optimizer weights/history are excluded after adaptation."""
    n=8
    payload={'schema':'MA-446/inference-v1','experiment':'MA-446','world_seed':world_seed,
             'method':method,'task_ids':list(range(n))}
    if method=='hard_shared':
        payload['shared']={'w0':w0}
    elif method in ('full_adam','lstm_full'):
        payload['task_weights']=result['params'][:n]
    else:
        payload['shared']={'w0':w0,'basis':basis}
        payload['task_codes']=result['params'][:n]
    return payload


def save_inference_payload(path: Path, payload: dict[str, Any],
                           episode: dict[str, torch.Tensor], expected: torch.Tensor) -> dict[str, Any]:
    blob=tensor_bytes(payload); path.write_bytes(blob)
    restored=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
    if 'task_weights' in restored:
        weights=restored['task_weights']
        pred=torch.einsum('bni,boi->bno',episode['query_x'][:8],weights)
    elif 'task_codes' in restored:
        weights=decode_m(restored['shared']['w0'],restored['shared']['basis'],restored['task_codes'])
        pred=torch.einsum('bni,boi->bno',episode['query_x'][:8],weights)
    else:
        pred=torch.einsum('bni,oi->bno',episode['query_x'][:8],restored['shared']['w0'])
    diff=float((pred-expected).abs().max())
    if diff != 0.0: raise AssertionError(f'inference payload replay difference {diff}')
    return {'path':str(path.relative_to(ROOT)),'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'roundtrip_max_prediction_difference':diff}

def make_payload(method: str, world_seed: int, w0: torch.Tensor,
                 basis: torch.Tensor, result: dict[str, Any], episode: dict[str, torch.Tensor],
                 learned_model: CoordinateLSTM | None = None,
                 meta_sgd: tuple[torch.Tensor,torch.Tensor] | None = None,
                 lr: float | None = None) -> dict[str, Any]:
    n=8
    if method=='hard_shared':
        pred=torch.einsum('bni,oi->bno',episode['query_x'][:n],w0)
        return {'schema':'MA-446/v1','experiment':'MA-446','world_seed':world_seed,'method':method,'task_ids':list(range(n)),'adaptation_updates':STEPS,'optimizer_step':None,'shared':{'w0':w0},'states':{}}
    params=result['params'][:n]
    if method in ('full_adam','lstm_full'):
        pred=torch.einsum('bni,boi->bno',episode['query_x'][:n],params)
    else:
        pred=torch.einsum('bni,boi->bno',episode['query_x'][:n],decode_m(w0,basis,params))
    state={'params':params}
    if 'm1' in result: state.update({'m1':result['m1'][:n],'m2':result['m2'][:n]})
    if 'state' in result: state.update({'h':result['state'][0][:n],'c':result['state'][1][:n]})
    payload={'schema':'MA-446/v1','experiment':'MA-446','world_seed':world_seed,'method':method,'task_ids':list(range(n)),'adaptation_updates':STEPS,'optimizer_step':STEPS if method in ('mirror_adam','full_adam') else None,'shared':{},'states':state}
    if method not in ('full_adam','lstm_full'):
        payload['shared'].update({'w0':w0,'basis':basis})
    if learned_model is not None: payload['shared']['learned_optimizer_state_dict']=learned_model.state_dict()
    if meta_sgd is not None: payload['shared'].update({'meta_init':meta_sgd[0],'meta_alpha':meta_sgd[1]})
    if lr is not None: payload['hyperparameters']={'learning_rate':lr}
    return payload


def run_world(seed: int, artifacts: Path) -> dict[str, Any]:
    w0,basis=make_world(seed)
    train_m,wall_train_m,losses_m=train_lstm(seed,w0,basis,True,seed+31)
    train_w,wall_train_w,losses_w=train_lstm(seed,w0,basis,False,seed+131)
    for model in (train_m,train_w):
        for parameter in model.parameters(): parameter.requires_grad_(False)
    meta_init,meta_alpha,wall_meta,losses_meta=train_meta_sgd(seed,w0,basis)
    dev=make_episode(32,w0,basis,generator(seed+2_000_000))
    methods:dict[str,dict[str,Any]]={}
    qhard=query_mse_w(w0.unsqueeze(0).expand(32,-1,-1),dev['query_x'],dev['query_y'])
    methods['hard_shared']={'curve':[float(qhard.mean())]*5,'task_curves':[qhard.tolist()]*5,'params':None,'wall_seconds_per_task':0.0}
    # Tune one learning rate per optimizer across both worlds in the caller.
    for kind in ('sgd','adam'):
        for lr in LRS:
            key=f'{kind}_{lr:g}_mirror'
            methods[key]=adapt_sgd_or_adam(dev,w0,basis,True,kind,lr)
    methods['meta_sgd_mirror']=adapt_meta_sgd(dev,w0,basis,meta_init,meta_alpha)
    methods['lstm_mirror']=adapt_lstm(train_m,dev,w0,basis,True)
    for lr in LRS:
        methods[f'adam_{lr:g}_full']=adapt_sgd_or_adam(dev,w0,basis,False,'adam',lr)
    methods['lstm_full']=adapt_lstm(train_w,dev,w0,basis,False)
    train_seconds={'lstm_mirror':wall_train_m,'lstm_full':wall_train_w,'meta_sgd_mirror':wall_meta}
    chosen={}
    # This helper call uses an explicit two-world pooled selection in main after both worlds run;
    # preliminary choices are updated later without changing any learned settings.
    for kind in ('sgd','adam'):
        chosen[kind]=min(LRS,key=lambda lr:methods[f'{kind}_{lr:g}_mirror']['curve'][-1])
    chosen['full_adam']=min(LRS,key=lambda lr:methods[f'adam_{lr:g}_full']['curve'][-1])
    return {'world_seed':seed,'w0':w0,'basis':basis,'dev':dev,'methods':methods,'chosen':chosen,
            'lstm_mirror':train_m,'lstm_full':train_w,'meta_init':meta_init,'meta_alpha':meta_alpha,
            'meta_train_seconds':train_seconds,'meta_train_loss_final':{'lstm_mirror':losses_m[-1],'lstm_full':losses_w[-1],'meta_sgd_mirror':losses_meta[-1]}}
