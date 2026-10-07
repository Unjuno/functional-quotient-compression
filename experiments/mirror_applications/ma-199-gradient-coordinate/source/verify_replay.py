import csv
import sys
from pathlib import Path

import torch
import run

# Timing varies; replay the deterministic model, payload and compute metrics.
run.throughput = lambda *args, **kwargs: 0.0


def read(paths):
    combined = []
    for path in paths:
        combined.extend(csv.DictReader(Path(path).open()))
    return combined


def check(paths):
    input_rows = read(paths); unique = {}
    key_fields = ('split', 'seed', 'condition', 'learning_rate', 'method', 'stage')
    for row in input_rows:
        key = tuple(row[k] for k in key_fields)
        if key in unique: assert unique[key] == row, key
        unique[key] = row
    groups = {}
    for row in unique.values():
        groups.setdefault((row['split'], int(row['seed']), row['condition'], float(row['learning_rate'])), []).append(row)
    exact = ('shared_base_bytes', 'inference_payload_bytes', 'incremental_inference_bytes', 'resume_payload_bytes',
             'train_examples_cumulative', 'optimizer_updates_cumulative', 'active_compute_proxy')
    metrics = ('mean_seen_mse', 'forgetting_abs', 'task0_mse', 'task1_mse', 'task2_mse', 'task3_mse', 'task4_mse')
    checked, maximum = 0, 0.0
    for (split, seed, condition, lr), saved in groups.items():
        replay = run.run_world(seed, condition, lr, split)
        saved = sorted(saved, key=lambda x: (int(x['stage']), x['method']))
        replay = sorted(replay, key=lambda x: (int(x['stage']), x['method']))
        assert len(saved) == len(replay)
        for old, new in zip(saved, replay):
            assert old['method'] == new['method'] and int(old['stage']) == new['stage']
            for field in exact: assert old[field] == str(new[field]), (field, old[field], new[field])
            for field in metrics:
                if old[field] and new[field]:
                    delta = abs(float(old[field]) - float(new[field])); maximum = max(maximum, delta)
                    assert delta <= 1e-9, (field, old[field], new[field])
            checked += 1
    return len(input_rows), len(unique), checked, maximum


if __name__ == '__main__':
    torch.set_num_threads(1)
    print(check(sys.argv[1:]))
