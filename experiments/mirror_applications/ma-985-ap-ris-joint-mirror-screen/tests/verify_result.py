"""Verify MA-985 power limits, serialization replay, and native-product alias."""
import csv,json,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parents[1]
p=json.loads((R/'PROTOCOL.json').read_text());rows=list(csv.DictReader((R/'RESULTS_CORE.csv').open()));fresh=json.loads((R/'source/FRESH.json').read_text())
assert p['experiment_id']=='MA-985' and p['protocol_frozen'] is True
assert len(rows)==15 and len(fresh)==3
assert {int(x['seed']) for x in rows}=={98511,98513,98517}
assert {x['condition'] for x in rows}=={'full','ap_low_fullris','fullap_ris_low','mirror','native_product'}
for w in fresh:
 by={x['condition']:x for x in w['results']};a=by['mirror'];b=by['native_product']
 assert a['payload_sha256']==b['payload_sha256'] and a['payload_bytes']==b['payload_bytes']
 assert a['mean_wsr']==b['mean_wsr'] and a['max_power_residual']<1e-5
 assert all(x['max_power_residual']<1e-5 for x in w['results'])
spec=importlib.util.spec_from_file_location('run_ma985',R/'source/run_ma985.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
replay=mod.evaluate(98511,2,2);saved={x['condition']:x for x in fresh[0]['results']}
for x in replay:
 assert abs(x['mean_wsr']-saved[x['condition']]['mean_wsr'])<1e-10
 assert x['payload_sha256']==saved[x['condition']]['payload_sha256']
assert replay[3]['payload_bytes']>replay[0]['payload_bytes']
assert replay[3]['mean_wsr']<.9*replay[0]['mean_wsr']
assert (R/'RESULTS.md').exists()
print('MA-985 verification passed: 15 fresh rows, 3 worlds, WMMSE power constraints, decoded replay, and exact native-product alias.')
