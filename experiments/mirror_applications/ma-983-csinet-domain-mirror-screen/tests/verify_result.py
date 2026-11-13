"""Integrity and exact embedding-alias checks for MA-983."""
import csv,json,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parents[1]
p=json.loads((R/'PROTOCOL.json').read_text());rows=list(csv.DictReader((R/'RESULTS_CORE.csv').open()));fresh=json.loads((R/'source/FRESH.json').read_text())
assert p['experiment_id']=='MA-983' and p['protocol_frozen'] is True
assert len(rows)==18 and len(fresh)==3
assert {int(x['seed']) for x in rows}=={98311,98313,98317}
spec=importlib.util.spec_from_file_location('run_ma983',R/'source/run_ma983.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
by={(int(x['seed']),x['condition']):x for x in rows}
for seed in (98311,98313,98317):
  replay,_=mod.world(seed);q={x['condition']:x for x in replay}
  assert q['mirror']['payload_sha256']==q['embedding']['payload_sha256']
  assert q['mirror']['bytes']==q['embedding']['bytes']==26781
  for metric,csv_metric in [('mean_nmse_db','mean_nmse_db'),('mean_rate','mean_spectral_efficiency')]:
   assert abs(q['mirror'][metric]-q['embedding'][metric])<1e-12
   assert abs(q['mirror'][metric]-float(by[(seed,'mirror')][csv_metric]))<1e-10
  assert q['mirror']['feedback_bits']==q['independent']['feedback_bits']==130
  assert q['mirror']['mean_rate']<.95*q['independent']['mean_rate']
assert (R/'RESULTS.md').exists()
print('MA-983 verification passed: 18 fresh rows, 3 held-out domain worlds, decoded payload replay; Mirror and matched embedding are byte/function identical.')
