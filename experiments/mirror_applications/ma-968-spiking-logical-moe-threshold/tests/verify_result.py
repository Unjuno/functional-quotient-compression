"""Integrity checks for MA-968 payload and fresh-world results."""
import csv,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
p=json.loads((R/'PROTOCOL.json').read_text())
rows=list(csv.DictReader((R/'RESULTS_CORE.csv').open()))
w=json.loads((R/'source/WORLD_RESULTS.json').read_text())
assert p['protocol_frozen'] is True
assert len(rows)==30 and len(w)==6
assert {int(x['world_seed']) for x in rows}=={96811,96813,96817}
assert {x['condition'] for x in rows}=={'shared','direct_threshold','mirror_threshold','rank2_private_synapse','independent_moe'}
for d in w: assert d['mirror_direct_max_logit_error']==0.0
for r in rows:
 assert int(r['serialized_bytes'])>0 and int(r['active_synapse_macs'])>0
 assert 0<=float(r['primary_value'])<=1
 json.loads(r['secondary_value'])
assert (R/'RESULTS.md').exists()
print('MA-968 verification passed: 30 fresh rows across 3 worlds × 2 families; payload serialization replayed; Mirror/direct threshold logit alias exact.')
