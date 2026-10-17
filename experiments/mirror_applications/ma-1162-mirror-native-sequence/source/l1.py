"""MA-1162 L1: frozen CPU synthetic causal query training harness.

All tasks expose only causal token prefixes, and each model is trained from scratch
on each independent world. This is conditional query prediction, not corpus LM NLL.
"""
import argparse
import csv
import io
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

VOCAB, WIDTH, SEQLEN, HIDDEN = 64, 32, 32, 64
TASKS = ('copy', 'retrieve', 'ordered', 'hidden')
MODELS = ('local', 'native', 'tied', 'gain4', 'rank1_4', 'mirror4', 'mirror4_t2', 'independent4')


def rotate_pairs(x, angle):
    """Orthonormal repeated Givens rotation, one angle for every 2-plane."""
    a, b = x[..., 0::2], x[..., 1::2]
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c * a - s * b, s * a + c * b), dim=-1).flatten(-2)


def make_world(world):
    g = torch.Generator().manual_seed(int(world))
    return torch.randperm(VOCAB, generator=g)


def make_batch(task, batch, generator, world_perm, length=SEQLEN):
    if task == 'copy':
        prefix = torch.randint(0, 32, (batch, length - 1), generator=generator)
        lag = torch.randint(1, 9, (batch,), generator=generator)
        y = prefix.gather(1, (length - 1 - lag).unsqueeze(1)).squeeze(1)
        x = torch.cat((prefix, (48 + lag).unsqueeze(1)), dim=1)
    elif task == 'retrieve':
        keys = torch.rand(batch, 16, generator=generator).argsort(-1)
        values = torch.randint(0, 3, (batch, 16), generator=generator)
        query_index = torch.randint(0, 16, (batch,), generator=generator)
        query_key = keys.gather(1, query_index[:, None]).squeeze(1)
        y = values.gather(1, query_index[:, None]).squeeze(1)
        first16 = 3 * keys + values
        pads = torch.full((batch, length - 17), 63, dtype=torch.long)
        x = torch.cat((first16, pads, (48 + query_key)[:, None]), dim=1)
    elif task == 'ordered':
        ops = torch.randint(0, 6, (batch, 6), generator=generator)
        state = torch.zeros(batch, dtype=torch.long)
        for j in range(ops.size(1)):
            state = (((ops[:, j] // 3) + 1) * state + ops[:, j] % 3) % 3
        y = state
        x = torch.cat((ops, torch.full((batch, length - 7), 63, dtype=torch.long),
                       torch.full((batch, 1), 60, dtype=torch.long)), dim=1)
    elif task == 'hidden':
        prefix = torch.randint(0, 32, (batch, length - 1), generator=generator)
        x = torch.cat((prefix, torch.full((batch, 1), 62, dtype=torch.long)), dim=1)
        # Outcome deliberately independent of the complete observed prefix.
        y = torch.randint(0, 3, (batch,), generator=generator)
    else:
        raise ValueError(task)
    return world_perm[x], y


class QueryNet(nn.Module):
    def __init__(self, kind, d=WIDTH, seq_len=SEQLEN):
        super().__init__()
        if kind not in MODELS:
            raise ValueError(kind)
        self.kind, self.d, self.seq_len = kind, d, seq_len
        self.K = 4 if kind in ('gain4', 'rank1_4', 'mirror4', 'mirror4_t2', 'independent4') else 1
        self.T = 2 if kind in ('tied', 'mirror4_t2') else 1
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Parameter(torch.randn(seq_len, d) * .02)
        self.norm = nn.LayerNorm(d)
        self.norm_ff = nn.LayerNorm(d)
        if kind != 'independent4':
            self.wq = nn.Linear(d, d, bias=False)
            self.wk = nn.Linear(d, d, bias=False)
            self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))
        self.head = nn.Linear(d, VOCAB)
        if kind in ('mirror4', 'mirror4_t2'):
            self.q_code = nn.Parameter(torch.linspace(-.55, .55, self.K))
            self.k_code = nn.Parameter(torch.linspace(.55, -.55, self.K))
            self.v_code = nn.Parameter(torch.linspace(-.75, .75, self.K))
        elif kind == 'gain4':
            self.q_code = nn.Parameter(torch.linspace(-.20, .20, self.K))
            self.k_code = nn.Parameter(torch.linspace(.20, -.20, self.K))
            self.v_code = nn.Parameter(torch.linspace(-.20, .20, self.K))
        elif kind == 'rank1_4':
            self.q_adapter = nn.Parameter(torch.randn(self.K,d)*.02)
            self.k_adapter = nn.Parameter(torch.randn(self.K,d)*.02)
            self.v_adapter = nn.Parameter(torch.randn(self.K,d)*.02)
        elif kind == 'independent4':
            self.q_heads = nn.Parameter(torch.randn(self.K, d, d) / d**.5)
            self.k_heads = nn.Parameter(torch.randn(self.K, d, d) / d**.5)
            self.v_heads = nn.Parameter(torch.randn(self.K, d, d) / d**.5)
        # All K views are used in an equal-weight aggregate; no oracle routing.

    def forward(self, x, return_all=False):
        b, l = x.shape
        h = self.emb(x) + self.pos[:l]
        mask = torch.ones(l, l, device=x.device, dtype=torch.bool).triu(1)
        for _ in range(self.T):
            z = self.norm(h)
            if self.kind == 'local':
                h = h + self.ff(self.norm_ff(h))
                continue
            if self.kind == 'independent4':
                q = torch.einsum('bld,kde->bkle', z, self.q_heads)
                k = torch.einsum('bld,kde->bkle', z, self.k_heads)
                v = torch.einsum('bld,kde->bkle', z, self.v_heads)
            else:
                q0, k0, v0 = self.wq(z), self.wk(z), self.wv(z)
                if self.kind.startswith('mirror4'):
                    q = rotate_pairs(q0[:, None], self.q_code[None, :, None, None])
                    k = rotate_pairs(k0[:, None], self.k_code[None, :, None, None])
                    v = rotate_pairs(F.relu(rotate_pairs(v0[:, None], self.v_code[None, :, None, None])),
                                     -self.v_code[None, :, None, None])
                elif self.kind == 'rank1_4':
                    q = q0[:,None] + q0[:,None,:,0:1]*self.q_adapter[None,:,None,:]
                    k = k0[:,None] + k0[:,None,:,1:2]*self.k_adapter[None,:,None,:]
                    v = F.relu(v0)[:,None] + v0[:,None,:,1:2]*self.v_adapter[None,:,None,:]
                elif self.kind == 'gain4':
                    q = q0[:, None] * torch.exp(self.q_code[None, :, None, None])
                    k = k0[:, None] * torch.exp(self.k_code[None, :, None, None])
                    v = F.relu(v0[:, None] * torch.exp(self.v_code[None, :, None, None]))
                else:
                    q, k, v = q0[:, None], k0[:, None], v0[:, None]
            score = (q @ k.transpose(-1, -2)) / math.sqrt(self.d)
            attn = torch.softmax(score.masked_fill(mask[None, None], -torch.inf), dim=-1)
            mixed = (attn @ v).mean(dim=1)
            h = h + self.wo(mixed)
            h = h + self.ff(self.norm_ff(h))
        return self.head(h) if return_all else self.head(h[:, -1, :])


def payload_bytes(model):
    b = io.BytesIO()
    torch.save({'config': {'kind': model.kind, 'd': model.d, 'seq_len': model.seq_len},
                'state': model.state_dict()}, b)
    return len(b.getvalue())


def param_counts(model):
    physical = sum(p.numel() for p in model.parameters())
    codes = sum(p.numel() for n, p in model.named_parameters() if n.endswith('_code'))
    return physical, codes


def proxy_macs(model, length=SEQLEN):
    """Per example analytical dense MACs; excludes softmax, normalization, GELU, view ops."""
    d, k, t = model.d, model.K, model.T
    attention = 0 if model.kind == 'local' else (2 * k * length * length * d)
    proj = 0 if model.kind == 'local' else (3 * length * d*d * (k if model.kind == 'independent4' else 1) + length * d*d)
    ff = 4 * length * d*d
    # Include output head and a conservative embedding addition; no embedding lookup MAC.
    return t * (attention + proj + ff) + d*VOCAB


def evaluate(model, task, perm, g, n=384, batch=64):
    model.eval()
    total_loss, correct = 0., 0
    with torch.no_grad():
        for start in range(0, n, batch):
            x, y = make_batch(task, min(batch, n-start), g, perm)
            logits = model(x)
            total_loss += F.cross_entropy(logits, y, reduction='sum').item()
            correct += (logits.argmax(-1) == y).sum().item()
    return total_loss/n, correct/n


def run_one(task, kind, world, seed, updates, batch, lr, eval_n):
    torch.manual_seed(10000+seed)
    random.seed(10000+seed)
    model = QueryNet(kind)
    perm = make_world(world)
    train_gen = torch.Generator().manual_seed(world*11+seed*103+TASKS.index(task)*10101+1)
    dev_gen = torch.Generator().manual_seed(world*11+seed*103+TASKS.index(task)*10101+9999999)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.)
    started = time.perf_counter()
    model.train()
    for step in range(updates):
        x, y = make_batch(task, batch, train_gen, perm)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    train_s = time.perf_counter()-started
    nll, acc = evaluate(model, task, perm, dev_gen, n=eval_n)
    # Repeated isolated forwards with exact same fixture after training.
    tgen = torch.Generator().manual_seed(world*31+7)
    sample,_ = make_batch(task, 8, tgen, perm)
    model.eval()
    with torch.no_grad():
        for _ in range(3): model(sample)
        a = time.perf_counter()
        for _ in range(12): model(sample)
        infer_ms = (time.perf_counter()-a)*1000/12
    phys, codes = param_counts(model)
    return {'task':task,'model':kind,'world':world,'seed':seed,'updates':updates,
            'examples':updates*batch,'tokens':updates*batch*SEQLEN,
            'nll':round(nll,7),'accuracy':round(acc,7),
            'physical_params':phys,'mirror_code_params':codes,
            'virtual_projection_weights':3*model.K*WIDTH*WIDTH,
            'serialized_bytes':payload_bytes(model),
            'forward_mac_proxy_per_example':proxy_macs(model),
            'train_mac_proxy_3x':3*updates*batch*proxy_macs(model),
            'train_seconds':round(train_s,4),'infer_ms_batch8':round(infer_ms,5),
            'torch':torch.__version__,'cpu_threads':torch.get_num_threads(),
            'backend':'eager_cpu_fp32','length':SEQLEN,'batch':batch,'lr':lr}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['dev','fresh'],required=True)
    p.add_argument('--tasks',default=','.join(TASKS))
    p.add_argument('--models',default=','.join(MODELS))
    p.add_argument('--updates',type=int,default=96)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--eval-n',type=int,default=384)
    p.add_argument('--budget-mode',choices=['fixed','mac_match_mirror4'],default='fixed')
    p.add_argument('--worlds',default='618200')
    p.add_argument('--seeds',default='0')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    worlds=list(map(int,a.worlds.split(',')))
    seeds=list(map(int,a.seeds.split(',')))
    if a.phase=='dev':
        assert set(worlds)=={618200}
    else:
        assert set(worlds)=={618201,618202,618203}, 'fresh requires all frozen worlds'
    rows=[]
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        raise FileExistsError(f'Never overwrite existing evidence: {out}')
    for task in a.tasks.split(','):
        for kind in a.models.split(','):
            for world in worlds:
                for seed in seeds:
                    actual_updates=a.updates
                    if a.budget_mode=='mac_match_mirror4':
                        actual_updates=max(1,int(a.updates*proxy_macs(QueryNet('mirror4'))/proxy_macs(QueryNet(kind))))
                    row=run_one(task,kind,world,seed,actual_updates,a.batch,a.lr,a.eval_n)
                    row['budget_mode']=a.budget_mode
                    rows.append(row)
                    print(f'{task:8} {kind:14} {world} {seed} loss={row["nll"]:.4f} acc={row["accuracy"]:.3f} '+
                          f'bytes={row["serialized_bytes"]} sec={row["train_seconds"]:.2f}',flush=True)
                    with out.open('w',newline='') as f:
                        wr=csv.DictWriter(f,fieldnames=list(rows[0].keys()));wr.writeheader();wr.writerows(rows)
    meta={'phase':a.phase,'args':vars(a),'host':platform.platform(),'torch':torch.__version__,
          'cpu':platform.processor(),'threads':torch.get_num_threads(),'rows':len(rows)}
    out.with_suffix('.meta.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__':
    main()
