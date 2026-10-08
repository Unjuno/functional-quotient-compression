"""MA-403: reproducible CPU causal token-conditioning mechanism screen.

All training is on synthetic teacher probabilities. No teacher parameter/code
is passed to the learner. The shared frozen prefix may be cached for training;
latency measures a complete uncached sequence forward. See PROTOCOL.json.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import struct
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['none', 'static_mirror', 'local_mirror', 'mirror', 'film',
           'complex_coeff', 'pair_linear', 'rank2', 'full_matrix']
MAGIC = b'MA403V1\n'


def default_config() -> dict[str, Any]:
    return dict(vocab=24, width=24, heads=4, ffn=48, sequence_length=12,
                generator_hidden=8, rank=2, dtype='float32', device='cpu',
                threads=1, dropout=0.0)


def rotate_pairs(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Apply independent real plane rotations; angle supports broadcasting."""
    if x.shape[-1] % 2 or angle.shape[-1] != x.shape[-1] // 2:
        raise ValueError('An even feature width and one angle per pair are required.')
    xp = x.reshape(*x.shape[:-1], -1, 2)
    a, b = xp[..., 0], xp[..., 1]
    c, s = angle.cos(), angle.sin()
    return torch.stack((c*a-s*b, s*a+c*b), dim=-1).flatten(-2)


class Backbone(nn.Module):
    def __init__(self, cfg: dict[str, Any], seed: int):
        super().__init__()
        self.cfg = dict(cfg)
        d, f, v = cfg['width'], cfg['ffn'], cfg['vocab']
        if d % cfg['heads'] or d % 2:
            raise ValueError('Width must be even and divisible by head count.')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(v, d)
            self.ln1 = nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3*d, bias=False)
            self.proj = nn.Linear(d, d, bias=False)
            self.ln2 = nn.LayerNorm(d)
            self.ff1 = nn.Linear(d, f)
            self.ff2 = nn.Linear(f, d)
            self.lnout = nn.LayerNorm(d)
            self.head = nn.Linear(d, v, bias=False)
            nn.init.normal_(self.embedding.weight, std=0.5)
            nn.init.normal_(self.head.weight, std=0.35)
        self.requires_grad_(False)

    def features(self, tokens: torch.Tensor) -> tuple[torch.Tensor, ...]:
        b, t = tokens.shape
        d, nheads = self.cfg['width'], self.cfg['heads']
        e = self.embedding(tokens)
        pos = torch.arange(t, dtype=e.dtype, device=e.device)[:, None]
        omega = torch.exp(-math.log(10000)*torch.arange(0, d, 2,
                          dtype=e.dtype, device=e.device)/d)
        pe = torch.stack(((pos*omega).sin(), (pos*omega).cos()), dim=-1).flatten(-2)
        x = e + 0.1*pe
        qkv = self.qkv(self.ln1(x)).reshape(b, t, 3, nheads, d//nheads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        att = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        att = att.transpose(1, 2).reshape(b, t, d)
        h = F.layer_norm(x+self.proj(att), (d,))
        denom = torch.arange(t, dtype=h.dtype, device=h.device).clamp_min(1).sqrt()
        prefix = torch.cat((torch.zeros_like(h[:, :1]), h.cumsum(1)[:, :-1]), dim=1)
        prefix = prefix/denom[None, :, None]
        c = F.layer_norm(prefix, (d,))
        local = F.layer_norm(e, (d,))
        return h, c, local

    def tail(self, modulated: torch.Tensor) -> torch.Tensor:
        z = modulated + self.ff2(F.gelu(self.ff1(self.ln2(modulated))))
        return self.head(self.lnout(z))


class Adapter(nn.Module):
    def __init__(self, cfg: dict[str, Any], mode: str, seed: int):
        super().__init__()
        if mode not in METHODS:
            raise ValueError(f'Unknown conditioning mode: {mode}')
        self.cfg, self.mode = dict(cfg), mode
        d, hidden, rank = cfg['width'], cfg['generator_hidden'], cfg['rank']
        outputs = dict(local_mirror=d//2, mirror=d//2, film=2*d,
                       complex_coeff=d, pair_linear=2*d, rank2=rank, full_matrix=d*d)
        self.code_count = outputs.get(mode, 0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if mode == 'static_mirror':
                self.angles = nn.Parameter(torch.zeros(d//2))
            elif mode != 'none':
                self.generator = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(),
                                               nn.Linear(hidden, self.code_count))
                nn.init.zeros_(self.generator[-1].weight)
                nn.init.zeros_(self.generator[-1].bias)
                if mode == 'rank2':
                    self.left = nn.Parameter(torch.randn(d, rank)/math.sqrt(d))
                    self.right = nn.Parameter(torch.randn(rank, d)/math.sqrt(d))

    def forward(self, h: torch.Tensor, context: torch.Tensor,
                local: torch.Tensor) -> torch.Tensor:
        mode, d = self.mode, self.cfg['width']
        if mode == 'none':
            return h
        if mode == 'static_mirror':
            return rotate_pairs(h, self.angles)
        code = self.generator(local if mode == 'local_mirror' else context)
        if mode in ('mirror', 'local_mirror'):
            return rotate_pairs(h, code)
        if mode == 'film':
            gain, bias = code.chunk(2, -1)
            return (1+gain)*h+bias
        if mode == 'rank2':
            return h + ((h @ self.right.T)*code) @ self.left.T
        xp = h.reshape(*h.shape[:-1], d//2, 2)
        if mode == 'complex_coeff':
            ab = code.reshape(*h.shape[:-1], d//2, 2)
            a, b = 1+ab[..., 0], ab[..., 1]
            return torch.stack((a*xp[..., 0]-b*xp[..., 1],
                                b*xp[..., 0]+a*xp[..., 1]), -1).flatten(-2)
        if mode == 'pair_linear':
            matrix = code.reshape(*h.shape[:-1], d//2, 2, 2)
            return h + (matrix @ xp.unsqueeze(-1)).squeeze(-1).flatten(-2)
        if mode == 'full_matrix':
            matrix = code.reshape(*h.shape[:-1], d, d)/math.sqrt(d)
            return h + (matrix @ h.unsqueeze(-1)).squeeze(-1)
        raise AssertionError(mode)


def make_teacher(cfg: dict[str, Any], world: str, seed: int) -> Adapter:
    modes = dict(rotation='mirror', affine='film', dense='full_matrix')
    teacher = Adapter(cfg, modes[world], seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+1)
        with torch.no_grad():
            nn.init.normal_(teacher.generator[0].weight, std=0.8/math.sqrt(cfg['width']))
            nn.init.normal_(teacher.generator[0].bias, std=0.15)
            strength = dict(rotation=1.0, affine=0.6, dense=0.9)[world]
            nn.init.normal_(teacher.generator[-1].weight,
                            std=strength/math.sqrt(cfg['generator_hidden']))
            nn.init.normal_(teacher.generator[-1].bias, std=0.08)
    return teacher.requires_grad_(False).eval()


def heldout_mask(tokens: torch.Tensor) -> torch.Tensor:
    groups = tokens % 4
    result = torch.zeros_like(tokens, dtype=torch.bool)
    a, b = groups[:, :-1], groups[:, 1:]
    result[:, 1:] = ((a == 0) & (b == 1)) | ((a == 2) & (b == 3))
    return result


def make_tokens(n: int, length: int, vocab: int, seed: int,
                avoid: bool) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(vocab, (n, length), generator=generator)
    if avoid:
        for t in range(1, length):
            for _ in range(100):
                a, b = tokens[:, t-1] % 4, tokens[:, t] % 4
                bad = ((a == 0) & (b == 1)) | ((a == 2) & (b == 3))
                if not bad.any():
                    break
                tokens[bad, t] = torch.randint(vocab, (int(bad.sum()),), generator=generator)
            else:
                raise RuntimeError('Rejection sampler did not terminate.')
    return tokens


@torch.no_grad()
def cache_data(backbone: Backbone, teacher: Adapter, tokens: torch.Tensor,
               mask: torch.Tensor) -> dict[str, torch.Tensor]:
    h, c, local = backbone.features(tokens)
    logp = F.log_softmax(backbone.tail(teacher(h, c, local)), -1)
    return dict(h=h, c=c, local=local, logp=logp, q=logp.exp(),
                mask=mask, tokens=tokens)


def forward(backbone: Backbone, adapter: Adapter, tokens: torch.Tensor) -> torch.Tensor:
    return backbone.tail(adapter(*backbone.features(tokens)))


@torch.no_grad()
def metrics(backbone: Backbone, adapter: Adapter,
            data: dict[str, torch.Tensor]) -> dict[str, float]:
    total, ce, kl, entropy, correct = 0, 0.0, 0.0, 0.0, 0
    for start in range(0, len(data['h']), 128):
        s = slice(start, start+128)
        logp = F.log_softmax(backbone.tail(adapter(data['h'][s], data['c'][s],
                                                   data['local'][s])), -1).double()
        q, target, mask = data['q'][s].double(), data['logp'][s].double(), data['mask'][s]
        ce += float((-(q*logp).sum(-1))[mask].sum())
        kl += float(((q*(target-logp)).sum(-1))[mask].sum())
        entropy += float((-(q*target).sum(-1))[mask].sum())
        correct += int(((logp.argmax(-1) == q.argmax(-1)) & mask).sum())
        total += int(mask.sum())
    if not total:
        raise ValueError('Evaluation split contains no scored tokens.')
    return dict(cross_entropy=ce/total, excess_kl=max(0.0, kl/total),
                teacher_entropy=entropy/total, top1_agreement=correct/total,
                scored_tokens=total)


def fit(backbone: Backbone, adapter: Adapter, data: dict[str, torch.Tensor],
        updates: int, batch: int, lr: float, seed: int,
        record_every: int = 200) -> dict[str, Any]:
    params = list(adapter.parameters())
    if not params:
        return dict(updates=0, elapsed_s=0.0, curve=[])
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    curve = []
    begin = time.perf_counter()
    for step in range(updates):
        idx = torch.randint(len(data['h']), (batch,), generator=generator)
        rate = lr*(0.1+0.9*(1+math.cos(math.pi*step/max(1,updates-1)))/2)
        optimizer.param_groups[0]['lr'] = rate
        optimizer.zero_grad(set_to_none=True)
        mod = adapter(data['h'][idx], data['c'][idx], data['local'][idx])
        logp = F.log_softmax(backbone.tail(mod), -1)
        loss = -(data['q'][idx]*logp).sum(-1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {step}.')
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step == 0 or (step+1) % record_every == 0 or step+1 == updates:
            curve.append(dict(step=step+1, minibatch_ce=float(loss.detach())))
    return dict(updates=updates, elapsed_s=time.perf_counter()-begin, curve=curve)


def save_model(path: Path, backbone: Backbone, adapter: Adapter) -> dict[str, Any]:
    """Deterministic, pickle-free float32 archive with self-describing metadata."""
    tensors, raw, offset = [], [], 0
    states = {**{'backbone.'+k:v for k,v in backbone.state_dict().items()},
              **{'adapter.'+k:v for k,v in adapter.state_dict().items()}}
    for name, value in sorted(states.items()):
        a = np.ascontiguousarray(value.detach().cpu().numpy(), dtype='<f4')
        body = a.tobytes()
        tensors.append(dict(name=name, shape=list(a.shape), dtype='<f4',
                            offset=offset, nbytes=len(body)))
        raw.append(body);offset += len(body)
    metadata = dict(version=1, config=backbone.cfg, mode=adapter.mode, tensors=tensors)
    header = json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode('utf-8')
    payload = MAGIC+struct.pack('<Q',len(header))+header+b''.join(raw)
    path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(payload)
    return dict(bytes=path.stat().st_size, tensor_bytes=offset,
                sha256=hashlib.sha256(payload).hexdigest(),
                adapter_parameters=sum(p.numel() for p in adapter.parameters()),
                backbone_parameters=sum(p.numel() for p in backbone.parameters()))


def load_model(path: Path) -> tuple[Backbone, Adapter]:
    payload = path.read_bytes()
    if payload[:8] != MAGIC:
        raise ValueError('Invalid inference archive magic.')
    header_length = struct.unpack('<Q', payload[8:16])[0]
    if header_length > len(payload)-16:
        raise ValueError('Truncated archive header.')
    meta = json.loads(payload[16:16+header_length])
    body = payload[16+header_length:]
    b, a = Backbone(meta['config'], 0), Adapter(meta['config'], meta['mode'], 0)
    states = dict(backbone={}, adapter={})
    for t in meta['tensors']:
        lo, hi = t['offset'], t['offset']+t['nbytes']
        if not (0 <= lo <= hi <= len(body)):
            raise ValueError('Invalid tensor extent.')
        arr = np.frombuffer(body[lo:hi], dtype=t['dtype']).copy().reshape(t['shape'])
        which, name = t['name'].split('.', 1)
        states[which][name] = torch.from_numpy(arr)
    b.load_state_dict(states['backbone'], strict=True)
    a.load_state_dict(states['adapter'], strict=True)
    return b.eval(), a.eval()


def tensor_hash(x: torch.Tensor) -> str:
    return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()


def build_world(protocol: dict[str, Any], seed: int, world: str):
    cfg = protocol['architecture']
    b = Backbone(cfg, seed)
    teacher = make_teacher(cfg, world, seed+100000)
    splits = {}
    for i, (name, count_key, avoid) in enumerate([
        ('train','train_sequences',True),('iid','iid_sequences',True),
        ('ood','ood_sequences',False)]):
        tokens = make_tokens(protocol['data'][count_key], cfg['sequence_length'],
                             cfg['vocab'], seed+200000+i, avoid)
        mask = heldout_mask(tokens) if name == 'ood' else torch.ones_like(tokens,dtype=torch.bool)
        splits[name] = cache_data(b,teacher,tokens,mask)
    return b, splits


def compute_proxy(cfg: dict[str, Any], adapter: Adapter) -> dict[str, int]:
    d, f, v, t = cfg['width'], cfg['ffn'], cfg['vocab'], cfg['sequence_length']
    base = 4*d*d + 2*d*f + d*v + 2*t*d
    generator = 0 if adapter.mode in ('none','static_mirror') else d*cfg['generator_hidden']+cfg['generator_hidden']*adapter.code_count
    transform = dict(none=0,static_mirror=2*d,local_mirror=2*d,mirror=2*d,
                     film=d,complex_coeff=2*d,pair_linear=2*d,
                     rank2=2*d*cfg['rank'],full_matrix=d*d)[adapter.mode]
    return dict(linear_mac_proxy_per_token=base+generator+transform,
                generator_mac_proxy_per_token=generator,
                generated_scalars_per_token=adapter.code_count)


@torch.inference_mode()
def benchmark(backbone: Backbone, adapters: dict[str, Adapter],
              tokens: torch.Tensor, settings: dict[str, Any], seed: int):
    samples, summaries = [], {}
    rng = np.random.default_rng(seed)
    for batch in settings['batches']:
        x = tokens[:batch]
        for mode in adapters:
            for _ in range(settings['warmup_calls']):
                forward(backbone, adapters[mode], x)
        times = {mode: [] for mode in adapters}
        for repeat in range(settings['repeats']):
            for mode in rng.permutation(list(adapters)):
                begin = time.perf_counter_ns()
                for _ in range(settings['calls_per_repeat']):
                    forward(backbone, adapters[mode], x)
                seconds = (time.perf_counter_ns()-begin)*1e-9/settings['calls_per_repeat']
                times[mode].append(seconds)
                samples.append(dict(method=mode,batch=batch,repeat=repeat,latency_s=seconds))
        for mode, values in times.items():
            arr = np.array(values);u = float(arr.std(ddof=1)/math.sqrt(len(arr)))
            summaries[(mode,batch)] = dict(latency_median_s=float(np.median(arr)),
                latency_min_s=float(arr.min()),latency_max_s=float(arr.max()),
                latency_repeat_u_s=u,latency_repeat_expanded_k2_s=2*u,
                tokens_per_s=batch*tokens.shape[1]/float(np.median(arr)))
    return summaries, samples


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError('Cannot write an empty result table.')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def environment() -> dict[str, Any]:
    cpu=Path('/proc/cpuinfo').read_text()
    clocks=[float(x.split(':')[1]) for x in cpu.splitlines() if x.startswith('cpu MHz')]
    return dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
        platform=platform.platform(),cpu_model=next(x.split(':')[1].strip() for x in cpu.splitlines() if x.startswith('model name')),
        logical_cpus=os.cpu_count(),affinity=sorted(os.sched_getaffinity(0)),
        cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        memory_limit_bytes=int(Path('/sys/fs/cgroup/memory.max').read_text()),
        clock_snapshot_mhz=clocks,clock_locked=False,torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),cuda_available=torch.cuda.is_available(),
        execution='CPU eager FP32, F.scaled_dot_product_attention(is_causal=True), dropout=0',
        mkldnn_enabled=torch.backends.mkldnn.enabled,quantization=None,compile=False)


def evaluate_gates(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    g=protocol['gates']; outcomes=[]
    seeds=sorted({int(r['seed']) for r in rows})
    for seed in seeds:
        for world in protocol['worlds']:
            block=[r for r in rows if int(r['seed'])==seed and r['world']==world]
            by={r['method']:r for r in block};m=by['mirror']
            upper_ok=all(by['full_matrix'][f'{s}_excess_kl']<=g['upper_learnability_max_kl'] for s in ['iid','ood'])
            best={s:min(by[k][f'{s}_cross_entropy'] for k in protocol['native_controls']) for s in ['iid','ood']}
            qualified=[k for k in protocol['native_controls'] if all(by[k][f'{s}_cross_entropy']<=best[s]+g['quality_tolerance_vs_best_native_nat_per_token'] for s in ['iid','ood'])]
            quality=all(m[f'{s}_cross_entropy']<=best[s]+g['quality_tolerance_vs_best_native_nat_per_token'] and m[f'{s}_excess_kl']<=g['absolute_max_kl'] for s in ['iid','ood'])
            dynamic=all(by['static_mirror'][f'{s}_cross_entropy']-m[f'{s}_cross_entropy']>=g['minimum_gain_vs_static_nat_per_token'] for s in ['iid','ood'])
            storage_ref=min(qualified,key=lambda k:by[k]['full_bytes']) if qualified else None
            full_ratio=m['full_bytes']/by[storage_ref]['full_bytes'] if storage_ref else None
            extra_ratio=m['incremental_bytes']/by[storage_ref]['incremental_bytes'] if storage_ref else None
            storage=bool(storage_ref and full_ratio<=g['maximum_full_bytes_ratio'] and extra_ratio<=g['maximum_incremental_bytes_ratio'])
            latency={batch: m[f'b{batch}_latency_median_s']/min(by[k][f'b{batch}_latency_median_s'] for k in qualified) if qualified else None for batch in protocol['timing']['batches']}
            runtime=bool(qualified and all(x<=g['maximum_latency_ratio'] for x in latency.values()))
            passed=upper_ok and quality and dynamic and storage and runtime
            outcomes.append(dict(seed=seed,world=world,upper_learnable=upper_ok,
                quality_pass=quality,dynamic_usefulness_pass=dynamic,storage_pass=storage,
                runtime_pass=runtime,qualified_native=qualified,storage_reference=storage_ref,
                full_bytes_ratio=full_ratio,incremental_bytes_ratio=extra_ratio,
                latency_ratios=latency,verdict='PASS' if passed else ('FAIL' if upper_ok else 'NOT_ESTABLISHED')))
    all_pass=all(x['verdict']=='PASS' for x in outcomes)
    verdict='PROMISING_DEV_ONLY' if all_pass else ('FAIL' if any(x['verdict']=='FAIL' for x in outcomes) else 'NOT_ESTABLISHED')
    return dict(id='MA-403',stage='development',verdict=verdict,
                fresh_authorized=all_pass,fresh_executed=False,world_outcomes=outcomes,
                interpretation='A failed broad gate does not refute all token-wise Mirror parameterizations. No natural-language or capacity result.')


def run_development(out: Path, protocol: dict[str, Any], only_seed: int|None=None,
                    only_world: str|None=None) -> None:
    out.mkdir(parents=True,exist_ok=True)
    raw_protocol=(ROOT/'PROTOCOL.json').read_bytes()
    expected=(ROOT/'PROTOCOL.sha256').read_text().split()[0]
    assert hashlib.sha256(raw_protocol).hexdigest()==expected,'Frozen protocol changed'
    (out/'ENVIRONMENT.json').write_text(json.dumps(environment(),indent=2)+'\n')
    all_rows, all_timings, datasets = [], [], []
    for seed in protocol['dev_seeds']:
        if only_seed is not None and seed!=only_seed:continue
        for world in protocol['worlds']:
            if only_world is not None and world!=only_world:continue
            backbone,data=build_world(protocol,seed,world)
            for split, values in data.items():
                datasets.append(dict(seed=seed,world=world,split=split,
                    tokens_sha256=tensor_hash(values['tokens']),
                    probabilities_sha256=tensor_hash(values['q']),scored_tokens=int(values['mask'].sum())))
            adapters,world_rows = {},[]
            base_bytes=None
            for method in protocol['methods']:
                adapter=Adapter(protocol['architecture'],method,seed+300000)
                train=protocol['train']
                record=fit(backbone,adapter,data['train'],updates=train['updates'],
                    batch=train['batch_sequences'],lr=train['learning_rate'],seed=seed+400000)
                folder=out/f'{seed}_{world}_{method}';folder.mkdir(exist_ok=True)
                (folder/'TRAINING.json').write_text(json.dumps(record,indent=2)+'\n')
                storage=save_model(folder/'inference.bin',backbone,adapter)
                if method=='none':base_bytes=storage['bytes']
                row=dict(seed=seed,world=world,method=method,updates=record['updates'],
                    train_seconds=record['elapsed_s'],full_bytes=storage['bytes'],
                    incremental_bytes=storage['bytes']-base_bytes,
                    tensor_bytes=storage['tensor_bytes'],adapter_parameters=storage['adapter_parameters'],
                    backbone_parameters=storage['backbone_parameters'],payload_sha256=storage['sha256'])
                for split in ['train','iid','ood']:
                    row.update({f'{split}_{k}':v for k,v in metrics(backbone,adapter,data[split]).items()})
                row.update(compute_proxy(protocol['architecture'],adapter))
                world_rows.append(row);adapters[method]=adapter
                print(json.dumps({k:row[k] for k in ['seed','world','method','iid_excess_kl','ood_excess_kl','full_bytes','train_seconds']}),flush=True)
            timings,samples=benchmark(backbone,adapters,data['iid']['tokens'],protocol['timing'],seed+500000)
            all_timings.extend([dict(seed=seed,world=world,**s) for s in samples])
            for row in world_rows:
                for batch in protocol['timing']['batches']:
                    row.update({f'b{batch}_{k}':v for k,v in timings[(row['method'],batch)].items()})
            all_rows.extend(world_rows)
            write_csv(out/'RESULTS_CORE.csv',all_rows)
            write_csv(out/'TIMING_SAMPLES.csv',all_timings)
            write_csv(out/'DATA_MANIFEST.csv',datasets)
    (out/'GATES.json').write_text(json.dumps(evaluate_gates(protocol,all_rows),indent=2)+'\n')
    (out/'RUN_MANIFEST.json').write_text(json.dumps(dict(protocol_sha256=expected,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rows=len(all_rows),fresh_worlds_generated=[],stage='development'),indent=2)+'\n')


def replay(out: Path, protocol: dict[str, Any], retrain: bool=False) -> dict[str, Any]:
    with (out/'RESULTS_CORE.csv').open() as f:rows=list(csv.DictReader(f))
    checked=0;max_delta=0.0;retrained=[];cache={}
    for row in rows:
        seed=int(row['seed']);world=row['world'];method=row['method']
        key=(seed,world)
        if key not in cache:cache[key]=build_world(protocol,seed,world)
        original,data=cache[key]
        path=out/f'{seed}_{world}_{method}'/'inference.bin'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['payload_sha256']
        assert path.stat().st_size==int(row['full_bytes'])
        backbone,adapter=load_model(path)
        for split in ['train','iid','ood']:
            for name,value in metrics(backbone,adapter,data[split]).items():
                delta=abs(value-float(row[f'{split}_{name}']));max_delta=max(max_delta,delta)
                assert delta<=1e-12,(seed,world,method,split,name,delta)
        checked+=1
        if retrain and method=='mirror' and world=='rotation':
            new=Adapter(protocol['architecture'],method,seed+300000)
            t=protocol['train'];fit(original,new,data['train'],updates=t['updates'],
                batch=t['batch_sequences'],lr=t['learning_rate'],seed=seed+400000)
            temp=out/f'replay_{seed}.bin';saved=save_model(temp,original,new)
            assert saved['sha256']==row['payload_sha256'], 'Full training replay did not match'
            temp.unlink();retrained.append(dict(seed=seed,world=world,method=method,sha256=saved['sha256']))
    result=dict(payloads_checked=checked,quality_scalars_checked=checked*3*5,
                max_abs_metric_error=max_delta,retrained_exact=retrained,
                timing_replay='Not asserted; timing samples are stochastic measurements.',
                fresh_worlds_generated=[])
    (out/'REPLAY.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['dev','replay'])
    parser.add_argument('--out',type=Path,default=ROOT/'runs'/'dev_v1')
    parser.add_argument('--seed',type=int);parser.add_argument('--world',choices=['rotation','affine','dense'])
    parser.add_argument('--retrain',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    if args.command=='dev':run_development(args.out,protocol,args.seed,args.world)
    else:print(json.dumps(replay(args.out,protocol,args.retrain),indent=2))

if __name__=='__main__':main()
