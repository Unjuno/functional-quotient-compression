import csv
from pathlib import Path
from engine import run
ROOT=Path(__file__).resolve().parents[1]
raw=list(csv.DictReader((ROOT/'FRESH_RAW.csv').open()))
for seed in (29711,29712,29713):
    replay=run(seed,'fresh',0.0,1)
    saved=[r for r in raw if int(r['seed'])==seed]
    assert len(replay)==len(saved)==50
    by={(r['condition'],r['method'],int(r['tasks_seen'])):r for r in saved}
    for row in replay:
        old=by[(row['condition'],row['method'],row['tasks_seen'])]
        assert row['payload_sha256']==old['payload_sha256']
        assert row['inference_payload_bytes']==int(old['inference_payload_bytes'])
        assert abs(row['mean_seen_mse']-float(old['mean_seen_mse']))<1e-14
        assert row['reconstruction_max_abs_diff']<3e-7
print('fresh metrics, serialized bytes and state hashes replay exactly')
