"""Verify packed payload replay and native-gradient alias for MA-984."""
import csv,json,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parents[1]
p=json.loads((R/'PROTOCOL.json').read_text());rows=list(csv.DictReader((R/'RESULTS_CORE.csv').open()));fresh=json.loads((R/'source/FRESH.json').read_text())
assert p['experiment_id']=='MA-984' and p['protocol_frozen'] is True
assert len(rows)==15 and len(fresh)==3
assert {int(r['seed']) for r in rows}=={98411,98413,98417}
spec=importlib.util.spec_from_file_location('run_ma984',R/'source/run_ma984.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
for seed in (98411,98413,98417):
 z,detail=mod.evaluate(seed);by={x['condition']:x for x in z}
 a=by['mirror_two_coordinate_5bit'];b=by['native_two_coordinate_phase_gradient_5bit'];upper=by['continuous_full_element_upper'];full=by['native_full_element_2bit']
 assert a['rates']==b['rates'] and a['bytes']==b['bytes']==176 and a['bits']==b['bits']==10
 assert a['mean_rate']>=.95*upper['mean_rate']
 assert a['bytes']<=.5*full['bytes']
 assert next(x for x in detail if x['condition']=='mirror_two_coordinate_5bit')['payload_bytes']==176
assert (R/'RESULTS.md').exists()
print('MA-984 verification passed: 15 fresh rows, 3 worlds, packed decoder replay; Mirror/native phase-gradient outputs are identical.')
