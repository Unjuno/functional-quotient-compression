#!/usr/bin/env python3
"""A5: fit fixed variants; write and lock weight files BEFORE touching test splits."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import multiprocessing as mp
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from a5 import KINDS, create, serialize, deserialize, make_world, eval_model, train_model

ROOT = Path(__file__).resolve().parent
FAMILIES = ('compositional', 'independent')


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_group(world: int, family: str, caps: tuple[int, ...], seeds: tuple[int, ...], steps: int, out: str):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = make_world(world, family)
    root = Path(out)
    rows = []
    for cap in caps:
        for seed in seeds:
            for kind in KINDS:
                torch.manual_seed(801111 + world * 701 + seed * 31 + KINDS.index(kind) * 19)
                model = create(kind, cap)
                start = time.perf_counter()
                last_train = train_model(model, data, steps, seed=4189111 + world * 107 + seed * 47)
                elapsed = time.perf_counter() - start
                dv = eval_model(model, data['val'])
                raw = serialize(model)
                if len(raw) > cap:
                    raise RuntimeError('byte cap violation')
                fname = f'{family}_w{world}_s{seed}_b{cap}_{kind}.a5w'
                with (root / 'models' / fname).open('xb') as f:
                    f.write(raw)
                if serialize(deserialize(raw)) != raw:
                    raise RuntimeError('serialization verification failed')
                rows.append(dict(family=family, world=world, seed=seed, cap=cap, kind=kind,
                                 width=model.width, bytes=len(raw), artifact=fname,
                                 sha256=hashlib.sha256(raw).hexdigest(),
                                 train_last_batch_mse=last_train, val_nrmse=dv['nrmse'],
                                 val_worst_task=dv['worst_task_nrmse'],
                                 train_seconds=elapsed))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--worlds', type=int, nargs='+', default=list(range(500, 506)))
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--caps', type=int, nargs='+', default=[1500, 3000])
    ap.add_argument('--steps', type=int, default=800)
    ap.add_argument('--jobs', type=int, default=3)
    a = ap.parse_args()
    if a.jobs < 1 or a.jobs > 5 or a.steps < 1 or len(a.worlds) != len(set(a.worlds)):
        raise ValueError('invalid protocol parameters')
    out = a.out
    out.mkdir(parents=True, exist_ok=False)
    (out / 'models').mkdir()
    filenames = ['PROTOCOL.md', 'a5.py', 'test_a5.py', 'train.py', 'audit.py']
    current_clock = []
    machine = 'unknown'
    if Path('/proc/cpuinfo').exists():
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                machine = line.split(':', 1)[1].strip()
            elif line.startswith('cpu MHz'):
                current_clock.append(float(line.split(':', 1)[1]))
    protocol = dict(worlds=a.worlds, families=FAMILIES, seeds=a.seeds, caps=a.caps, steps=a.steps,
                    batch=256, kinds=KINDS,
                    expected_models=len(a.worlds) * 2 * len(a.seeds) * len(a.caps) * len(KINDS),
                    source_sha256={f: digest(ROOT / f) for f in filenames},
                    environment=dict(cpu=machine, observed_mhz_range=[min(current_clock), max(current_clock)] if current_clock else None,
                                     clock_locked=False, visible_cpus=len(os.sched_getaffinity(0)),
                                     jobs=a.jobs, torch=torch.__version__, numpy=np.__version__,
                                     python=platform.python_version(), device='CPU', dtype='FP32'),
                    evidence='No IID/OOD model evaluations during training; exact stored models audited after train.lock.json')
    (out / 'protocol.json').write_text(json.dumps(protocol, sort_keys=True, indent=2) + '\n')
    start = time.perf_counter()
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.jobs, mp_context=mp.get_context('spawn')) as pool:
        fs = {pool.submit(run_group, w, fam, tuple(a.caps), tuple(a.seeds), a.steps, str(out)): (w, fam)
              for w in a.worlds for fam in FAMILIES}
        for i, fut in enumerate(concurrent.futures.as_completed(fs), 1):
            part = fut.result()
            rows.extend(part)
            print(f'TRAIN_GROUP {i}/{len(fs)} {fs[fut]} {len(part)} models elapsed={time.perf_counter()-start:.1f}s', flush=True)
    rows.sort(key=lambda d: (d['family'], d['world'], d['seed'], d['cap'], d['kind']))
    if len(rows) != protocol['expected_models']:
        raise RuntimeError('missing training variants')
    with (out / 'train.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    digests = {f.name: digest(f) for f in (out / 'models').glob('*.a5w')}
    if len(digests) != len(rows) or any(digests[row['artifact']] != row['sha256'] for row in rows):
        raise RuntimeError('model hash mismatch')
    lock = dict(protocol_sha256=digest(out / 'protocol.json'), train_sha256=digest(out / 'train.csv'),
                models=digests, total_seconds=time.perf_counter() - start)
    (out / 'train.lock.json').write_text(json.dumps(lock, sort_keys=True, indent=2) + '\n')
    print('TRAIN_LOCKED', len(rows), 'models', 'duration_seconds', round(lock['total_seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
