import csv
import sys
from pathlib import Path

import torch

import run

# Replay checks deterministic metrics only; omit repeated wall/throughput screens.
run.throughput = lambda *args, **kwargs: 0.0
run_world = run.run_world


def check(path):
    rows = list(csv.DictReader(Path(path).open()))
    grouped = {}
    for row in rows:
        grouped.setdefault((row['split'], int(row['seed']), row['condition'], float(row['learning_rate'])), []).append(row)
    checked = 0
    max_mse_diff = 0.0
    exact_fields = ('shared_backbone_bytes', 'inference_payload_bytes', 'incremental_inference_bytes',
                    'resume_payload_bytes', 'train_examples_cumulative', 'optimizer_updates_cumulative',
                    'active_compute_proxy')
    metric_fields = ('mean_seen_mse', 'forgetting_abs', 'task0_mse', 'task1_mse', 'task2_mse', 'task3_mse', 'task4_mse')
    for (split, seed, condition, lr), saved in grouped.items():
        replay = run_world(seed, condition, lr, split)
        assert len(saved) == len(replay)
        for old, new in zip(sorted(saved, key=lambda x: int(x['stage'])), sorted(replay, key=lambda x: int(x['stage']))):
            assert old['method'] == new['method'] and int(old['stage']) == new['stage']
            for field in exact_fields:
                assert old[field] == str(new[field]), (field, old[field], new[field])
            for field in metric_fields:
                if old[field] and new[field]:
                    delta = abs(float(old[field]) - float(new[field]))
                    max_mse_diff = max(max_mse_diff, delta)
                    assert delta <= 1e-8, (field, old[field], new[field])
            checked += 1
    return len(rows), checked, max_mse_diff


if __name__ == '__main__':
    torch.set_num_threads(1)
    for result_path in sys.argv[1:]:
        print(result_path, check(result_path))
