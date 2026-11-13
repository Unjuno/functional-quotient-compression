"""Integrity checks for the preserved MA-980 invalid byte-attribution screen."""
import csv,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
p=json.loads((R/'PROTOCOL.json').read_text())
rows=list(csv.DictReader((R/'RESULTS_CORE.csv').open()))
fresh=json.loads((R/'source/FRESH.json').read_text())
assert p['experiment_id']=='MA-980' and p['protocol_frozen'] is True
assert len(rows)==96 and len(fresh)==3
assert {int(r['seed']) for r in rows}=={98011,98013,98017}
assert {r['condition'] for r in rows}=={'shared','direct_phase','mirror_phase','independent_dense'}
for seed in (98011,98013,98017):
 by={(r['condition'],int(r['task'])):r for r in rows if int(r['seed'])==seed}
 for task in range(8):
  assert float(by[('direct_phase',task)]['relative_frobenius_error'])==0.0
  assert float(by[('mirror_phase',task)]['relative_frobenius_error'])==0.0
  assert int(by[('mirror_phase',task)]['payload_bytes'])<int(by[('direct_phase',task)]['payload_bytes'])
assert 'different field names' in p['post_run_audit']
assert (R/'RESULTS.md').exists()
print('MA-980 artifact checks passed: 96 fresh rows, 3 worlds, exact decoded phase outputs; serialization byte confound disclosed.')
