"""Check fresh metrics and the native-vs-Mirror functional alias."""
import csv,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
p=json.loads((R/'PROTOCOL.json').read_text())
rows=list(csv.DictReader((R/'RESULTS_CORE.csv').open()))
fresh=json.loads((R/'source/FRESH.json').read_text())
assert p['experiment_id']=='MA-981' and p['protocol_frozen'] is True
assert len(rows)==15 and len(fresh)==3
assert {int(x['seed']) for x in rows}=={98111,98113,98117}
for seed in (98111,98113,98117):
 z={x['condition']:x for x in rows if int(x['seed'])==seed}
 a=z['mirror_sector_phase_view'];b=z['native_sector_steering_codebook']
 for metric in ('mean_spectral_efficiency','5th_percentile_spectral_efficiency','payload_bytes','feedback_bits_per_beam','complex_macs_per_user','phase_ops_per_user'):
  assert a[metric]==b[metric]
 assert a['phase_ops_per_user']=='64'
 assert float(a['mean_spectral_efficiency'])>=.95*float(z['independent_sector_dense_codebooks']['mean_spectral_efficiency'])
 assert int(a['payload_bytes'])==int(b['payload_bytes'])
assert (R/'RESULTS.md').exists()
print('MA-981 verification passed: 15 fresh rows across 3 channel worlds; Mirror/native rate, bytes and operation proxies match exactly.')
