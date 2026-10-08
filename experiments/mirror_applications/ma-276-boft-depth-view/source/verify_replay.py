import csv
from pathlib import Path
from engine import run
ROOT=Path(__file__).resolve().parents[1]
saved=list(csv.DictReader((ROOT/'FRESH_RAW.csv').open()))
for seed in (27611,27612,27613):
 rows=run(seed,'fresh',4);assert len(rows)==14
 table={(r['condition'],r['method']):r for r in saved if int(r['seed'])==seed}
 for r in rows:
  q=table[(r['condition'],r['method'])]
  assert r['payload_sha256']==q['payload_sha256']
  assert r['serialized_bytes']==int(q['serialized_bytes'])
  assert abs(r['composed_output_mse']-float(q['composed_output_mse']))<1e-14
  assert r['runtime_path_max_abs_diff']<3e-7
print('fresh MSE, actual payload sizes and hashes replay exactly')
