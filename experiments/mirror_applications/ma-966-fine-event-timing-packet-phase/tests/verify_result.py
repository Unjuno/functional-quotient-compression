"""Integrity checks for the locked MA-966 result bundle."""
import csv, json, hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
prov=json.loads((ROOT/'source/DATA_PROVENANCE.json').read_text())
protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
rows=list(csv.DictReader((ROOT/'RESULTS_CORE.csv').open()))
fresh=json.loads((ROOT/'source/FRESH_RESULTS.json').read_text())
alias=json.loads((ROOT/'source/ALIAS_AUDIT.json').read_text())
assert protocol['protocol_frozen'] is True
assert prov['event_trials']==708 and prov['split_counts']=={'train_trials':202,'development_trials':101,'fresh_trials':405}
assert len(rows)==18 and len(fresh)==18
assert {r['condition'] for r in rows}=={'raw','periodic','mirror','offset','phaseonly','independent'}
assert {int(r['seed']) for r in rows}=={96611,96613,96617}
assert alias['max_abs_logit_error']==0.0
by={(r['condition'],int(r['seed'])):r for r in fresh}
for seed in (96611,96613,96617):
    a=by[('mirror',seed)];b=by[('offset',seed)]
    assert a['fresh_macro_accuracy']==b['fresh_macro_accuracy']
    assert a['fresh_accuracy_by_offset']==b['fresh_accuracy_by_offset']
for r in rows:
    assert int(r['serialized_bytes'])>0 and float(r['primary_value'])>=0 and float(r['primary_value'])<=1
    json.loads(r['secondary_value'])
assert (ROOT/'RESULTS.md').exists()
print('MA-966 verification passed: locked split counts; 18 fresh rows; exact phase/native-offset accuracy alias; serialized replay recorded.')
