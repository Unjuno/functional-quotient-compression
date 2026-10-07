import argparse
import csv
import json
import os
import torch
torch.set_num_threads(1)

from engine import METHODS, TASKS, world, teacher_logits, effective_teacher_state, Student, evaluate, mac_proxy


def task_data(w, seed, n):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(n, 12, generator=g), None) for _ in range(TASKS)]


def train_world(seed, condition, lr, updates):
    w = world(seed, condition)
    data = task_data(w, seed + 100000, 128)
    students = {m: Student(m, w['base'], seed + 300000, lr) for m in METHODS}
    rows = []
    for task, (x, _) in enumerate(data):
        target_logits = teacher_logits(x, {'teachers': w['teachers'], 'condition': condition, 'angles': w['angles']}, task)
        for method, student in students.items():
            state = effective_teacher_state(w, task)
            student.acquire(task, x, target_logits, lr, updates=updates, teacher_state=state)
            metrics = evaluate(student, w, seed + 200000 + task * 100, task)
            for prior_task, metric in enumerate(metrics):
                payload = student.serialize()
                rows.append({
                    'condition': condition, 'world_or_seed': seed, 'stage_task': task, 'evaluated_task': prior_task,
                    'method': method, 'learning_rate': lr, 'serialized_bytes': len(payload),
                    'incremental_bytes': len(payload) - student.base_bytes(), 'resume_bytes': student.resume_bytes(),
                    'train_examples': student.examples.get(task, 0), 'optimizer_updates': student.updates.get(task, 0),
                    'active_compute_proxy': mac_proxy(method, student.examples.get(task, 0)),
                    'wall_time_s': student.wall.get(task, 0.0), 'teacher_kl': metric['kl'],
                    'teacher_top1_agreement': metric['agreement'], 'ece': metric['ece'],
                })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', required=True, help='comma separated integer seeds')
    p.add_argument('--conditions', default='aligned,independent')
    p.add_argument('--lrs', default='0.003,0.01')
    p.add_argument('--updates', type=int, default=300)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(',')]
    conditions = args.conditions.split(',')
    lrs = [float(x) for x in args.lrs.split(',')]
    rows = []
    for condition in conditions:
        for seed in seeds:
            for lr in lrs:
                print(f'condition={condition} seed={seed} lr={lr}', flush=True)
                rows.extend(train_world(seed, condition, lr, args.updates))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'rows': len(rows), 'out': args.out}, sort_keys=True))


if __name__ == '__main__':
    main()
