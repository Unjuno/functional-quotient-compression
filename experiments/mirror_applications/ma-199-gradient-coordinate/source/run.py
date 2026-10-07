import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch

from engine import D, METHODS, O, TASKS, Learner, evaluate, mac_proxy, teacher, world

EXAMPLES, UPDATES = 64, 300
FIELDS = ['split', 'seed', 'condition', 'learning_rate', 'method', 'stage', 'tasks_seen',
          'task0_mse', 'task1_mse', 'task2_mse', 'task3_mse', 'task4_mse', 'mean_seen_mse', 'forgetting_abs',
          'shared_base_bytes', 'inference_payload_bytes', 'incremental_inference_bytes', 'inference_bytes_per_skill',
          'resume_payload_bytes', 'resume_bytes_per_skill', 'train_examples_cumulative', 'optimizer_updates_cumulative',
          'active_compute_proxy', 'basis_setup_time_s', 'wall_time_s', 'inference_examples_per_s']


def throughput(m, seed, task):
    g = torch.Generator().manual_seed(seed); x = torch.randn(64, D, generator=g)
    with torch.no_grad():
        for _ in range(5): m.forward(x, task)
        start = time.perf_counter()
        for _ in range(30): m.forward(x, task)
    return 64 * 30 / (time.perf_counter() - start)


def run_world(seed, condition, lr, split):
    w = world(seed, condition); rows = []
    for mi, method in enumerate(METHODS):
        learner = Learner(method, w['base_w'], w['base_b'], w['plane'], seed + mi * 701, lr)
        acquired = {0: 0.0}; mac_total = 0
        for task in range(1, TASKS):
            gen = torch.Generator().manual_seed(seed + task * 1009)
            x = torch.randn(EXAMPLES, D, generator=gen); y = teacher(x, w, task)
            learner.acquire(task, x, y, lr, UPDATES)
            mac_total += mac_proxy(method, learner.examples.get(task, 0))
            mse = evaluate(learner, w, seed + 49001, task)
            increases = [mse[t] - acquired[t] for t in range(task)]
            forgetting = sum(increases) / len(increases) if increases else 0.0
            for old_task in range(task): acquired[old_task] = mse[old_task]
            acquired[task] = mse[task]
            payload, base = learner.inference_bytes(), learner.base_bytes()
            resume, resume_base = learner.resume_bytes(), learner.resume_base_bytes()
            row = {'split': split, 'seed': seed, 'condition': condition, 'learning_rate': f'{lr:.3f}', 'method': method,
                   'stage': task, 'tasks_seen': task + 1, 'mean_seen_mse': f'{statistics.mean(mse):.10g}',
                   'forgetting_abs': f'{forgetting:.10g}', 'shared_base_bytes': base, 'inference_payload_bytes': payload,
                   'incremental_inference_bytes': payload - base, 'inference_bytes_per_skill': f'{(payload-base)/task:.6f}',
                   'resume_payload_bytes': resume, 'resume_bytes_per_skill': f'{max(0,resume-resume_base)/task:.6f}',
                   'train_examples_cumulative': sum(learner.examples.values()), 'optimizer_updates_cumulative': sum(learner.updates.values()),
                   'active_compute_proxy': mac_total, 'basis_setup_time_s': f"{w['basis_setup_time']:.8f}",
                   'wall_time_s': f'{sum(learner.wall.values()):.6f}',
                   'inference_examples_per_s': f'{throughput(learner,seed+task*37,task):.3f}'}
            for i in range(TASKS): row[f'task{i}_mse'] = f'{mse[i]:.10g}' if i <= task else ''
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(); p.add_argument('--split', choices=('development', 'fresh'), required=True)
    p.add_argument('--seeds', nargs='+', type=int, required=True); p.add_argument('--lr', type=float, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args(); torch.set_num_threads(1); rows = []
    for seed in a.seeds:
        for condition in ('aligned', 'independent'):
            got = run_world(seed, condition, a.lr, a.split); rows.extend(got)
            for r in got: print(a.split, seed, condition, r['method'], 'stage', r['stage'], r['mean_seen_mse'], r['inference_bytes_per_skill'], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'split': a.split, 'seeds': a.seeds, 'learning_rate': a.lr, 'rows': len(rows)}))


if __name__ == '__main__': main()
