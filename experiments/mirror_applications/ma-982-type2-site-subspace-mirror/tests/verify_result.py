"""Verify MA-982 fresh accounting and native/Mirror functional alias."""
import csv,json,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parents[1]
p=json.loads((R/'PROTOCOL.json').read_text());rows=list(csv.DictReader((R/'RESULTS_CORE.csv').open()))
fresh=json.loads((R/'source/FRESH.json').read_text())
assert p['experiment_id']=='MA-982' and p['protocol_frozen'] is True
assert len(rows)==15 and len(fresh)==3
assert {int(r['seed']) for r in rows}=={98211,98213,98217}
assert {r['condition'] for r in rows}=={'shared_global_subspace_no_site_code','mirror_shared_subspace_plus_site_phase_coordinate','native_phase_parametric_TypeII','stored_per_site_TypeII_subspace','full_quantized_CSI_feedback'}
spec=importlib.util.spec_from_file_location('run_ma982',R/'source/run_ma982.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
for seed in (98211,98213,98217):
 d={x['condition']:x for x in mod.evaluate(seed)}
 a=d['mirror_shared_subspace_plus_site_phase_coordinate'];b=d['native_phase_parametric_TypeII']
 for k in ('mean_rate','p5_rate','relative_nmse','bytes','feedback_bits','probes','macs'):assert a[k]==b[k]
 assert a['bytes']==2273 and a['feedback_bits']==34
 assert a['mean_rate']>=.95*d['full_quantized_CSI_feedback']['mean_rate']
 assert int(a['bytes'])*8+512*a['feedback_bits'] > .8*(int(d['stored_per_site_TypeII_subspace']['bytes'])*8+512*d['stored_per_site_TypeII_subspace']['feedback_bits'])
assert (R/'RESULTS.md').exists()
print('MA-982 verification passed: 15 fresh rows, 3 worlds, serialized replay recomputed; Mirror/native Type-II alias exact; total-description gate missed.')
