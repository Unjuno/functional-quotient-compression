import csv
from pathlib import Path
from engine import run
ROOT=Path(__file__).resolve().parents[1]
saved=list(csv.DictReader((ROOT/'FRESH_RAW.csv').open()))
for seed in (28811,28812,28813):
 rows=run(seed,'fresh',4);assert len(rows)==12
 table={(r['condition'],r['method']):r for r in saved if int(r['seed'])==seed}
 for r in rows:
  q=table[(r['condition'],r['method'])]
  assert r['payload_sha256']==q['payload_sha256']
  assert r['serialized_bytes']==int(q['serialized_bytes'])
  assert abs(r['task_output_mse']-float(q['task_output_mse']))<1e-14
  assert r['runtime_path_max_abs_diff']<3e-7
print('fresh metrics, serialized state bytes and hashes replay exactly')
