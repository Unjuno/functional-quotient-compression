import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch

from engine import D, METHODS, Learner, evaluate, mac_proxy, teacher, throughput, world

BUDGETS, UPDATES = (16, 64, 256), 200
FIELDS = ['split', 'seed', 'condition', 'learning_rate', 'budget_examples', 'method', 'heldout_task1_mse',
          'task0_mse_before', 'task0_mse_after', 'task0_mse_increase', 'shared_base_bytes', 'inference_payload_bytes',
          'incremental_inference_bytes', 'inference_bytes_per_skill', 'resume_payload_bytes', 'resume_bytes_per_skill',
          'train_examples', 'optimizer_updates', 'active_compute_proxy', 'wall_time_s', 'inference_examples_per_s']


def run_world(seed, condition, lr, split):
    w = world(seed, condition)
    gen = torch.Generator().manual_seed(seed + 1901)
    x_train = torch.randn(max(BUDGETS), D, generator=gen)
    y_train = teacher(x_train, w, 1)
    rows = []
    for mi, method in enumerate(METHODS):
        init_seed = seed + 7919 * mi
        for budget in BUDGETS:
            learner = Learner(method, w['base_w'], w['base_b'], init_seed, lr)
            start = time.perf_counter()
            learner.fit(x_train[:budget], y_train[:budget], UPDATES)
            wall = time.perf_counter() - start
            b0, b1, mse = evaluate(learner, w, seed + 44001)
            total = learner.inference_bytes(); base = learner.base_bytes()
            resume = learner.resume_bytes(); resume_base = learner.resume_base_bytes()
            rows.append({'split': split, 'seed': seed, 'condition': condition, 'learning_rate': f'{lr:.3f}',
                         'budget_examples': budget, 'method': method, 'heldout_task1_mse': f'{mse:.10g}',
                         'task0_mse_before': f'{b0:.10g}', 'task0_mse_after': f'{b1:.10g}',
                         'task0_mse_increase': f'{b1-b0:.10g}', 'shared_base_bytes': base,
                         'inference_payload_bytes': total, 'incremental_inference_bytes': total-base,
                         'inference_bytes_per_skill': f'{(total-base):.6f}', 'resume_payload_bytes': resume,
                         'resume_bytes_per_skill': f'{max(0,resume-resume_base):.6f}',
                         'train_examples': learner.examples, 'optimizer_updates': learner.updates,
                         'active_compute_proxy': mac_proxy(method, learner.examples), 'wall_time_s': f'{wall:.6f}',
                         'inference_examples_per_s': f'{throughput(learner,seed+budget):.3f}'})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=('development', 'fresh'), required=True)
    p.add_argument('--seeds', nargs='+', type=int, required=True)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(1); rows = []
    for seed in a.seeds:
        for condition in ('aligned', 'independent'):
            got = run_world(seed, condition, a.lr, a.split); rows.extend(got)
            for r in got:
                print(a.split, seed, condition, r['method'], r['budget_examples'], r['heldout_task1_mse'],
                      r['incremental_inference_bytes'], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n'); w.writeheader(); w.writerows(rows)
    print(json.dumps({'split': a.split, 'seeds': a.seeds, 'learning_rate': a.lr, 'rows': len(rows)}))


if __name__ == '__main__': main()
