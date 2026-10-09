import csv
import sys
from pathlib import Path

import torch
import run

# Replay deterministic model metrics and payload/compute accounting, not timing noise.
run.throughput = lambda *args, **kwargs: 0.0


def check(path):
    rows = list(csv.DictReader(Path(path).open()))
    groups = {}
    for row in rows:
        groups.setdefault((row['split'], int(row['seed']), row['condition'], float(row['learning_rate'])), []).append(row)
    exact = ('shared_base_bytes', 'inference_payload_bytes', 'incremental_inference_bytes', 'resume_payload_bytes',
             'train_examples', 'optimizer_updates', 'active_compute_proxy')
    metrics = ('heldout_task1_mse', 'task0_mse_before', 'task0_mse_after', 'task0_mse_increase')
    checked, maximum = 0, 0.0
    for (split, seed, condition, lr), saved in groups.items():
        replay = run.run_world(seed, condition, lr, split)
        saved = sorted(saved, key=lambda x: (int(x['budget_examples']), x['method']))
        replay = sorted(replay, key=lambda x: (int(x['budget_examples']), x['method']))
        assert len(saved) == len(replay)
        for old, new in zip(saved, replay):
            assert old['method'] == new['method'] and int(old['budget_examples']) == new['budget_examples']
            for field in exact: assert old[field] == str(new[field]), (field, old[field], new[field])
            for field in metrics:
                delta = abs(float(old[field]) - float(new[field])); maximum = max(maximum, delta)
                assert delta <= 1e-8, (field, old[field], new[field])
            checked += 1
    return len(rows), checked, maximum


if __name__ == '__main__':
    torch.set_num_threads(1)
    for path in sys.argv[1:]: print(path, check(path))
