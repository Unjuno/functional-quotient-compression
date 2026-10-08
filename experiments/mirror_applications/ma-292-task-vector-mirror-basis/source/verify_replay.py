import csv
from pathlib import Path
from engine import run
ROOT=Path(__file__).resolve().parents[1]
raw=list(csv.DictReader((ROOT/'FRESH_RAW.csv').open()))
for seed in (29211,29212,29213):
    replay=run(seed,'fresh',8)
    saved=[r for r in raw if int(r['seed'])==seed]
    assert len(replay)==len(saved)==12
    table={(r['condition'],r['method']):r for r in saved}
    for row in replay:
        old=table[(row['condition'],row['method'])]
        assert row['payload_sha256']==old['payload_sha256']
        assert row['serialized_bytes']==int(old['serialized_bytes'])
        assert abs(row['task_output_mse']-float(old['task_output_mse']))<1e-14
        assert abs(row['pair_arithmetic_mse']-float(old['pair_arithmetic_mse']))<1e-14
print('fresh metrics, payload bytes and SHA256 hashes replay exactly')
