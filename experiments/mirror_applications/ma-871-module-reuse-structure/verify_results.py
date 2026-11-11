"""Deterministic metric replay for MA-871; wall-clock fields are excluded."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
from experiment import run

saved=json.loads((ROOT/'fresh_results.json').read_text())
replay=run([907,919,937])
for key in ('mean_mse_by_support','max_mse_by_support','payload_bytes_at_8','mean_adaptation_seconds_at_8','mean_inference_seconds_at_8'):
    # Timing values are noisy; metrics and byte counts are deterministic.
    if 'seconds' in key:
        continue
    assert saved[key] == replay[key], key
for a,b in zip(saved['rows'], replay['rows']):
    assert a['seed']==b['seed']
    assert a['payload_bytes_at_8']==b['payload_bytes_at_8']
    for n in ('4','8','16','32'):
        x,y=a['by_support'][n],b['by_support'][n]
        assert x['mse']==y['mse']
        assert x['per_task_mse']==y['per_task_mse']
        assert x['payload_bytes']==y['payload_bytes']
        assert x['mse']['mirror']==x['mse']['native_pca']
print('MA-871 fresh metric replay: exact; native PCA alias: exact')
