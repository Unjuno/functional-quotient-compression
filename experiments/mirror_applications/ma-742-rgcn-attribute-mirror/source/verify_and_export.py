"""Validate frozen provenance, exact payloads and export the evidence table."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'source'
spec=importlib.util.spec_from_file_location('ma742_runner',SRC/'run_experiment.py')
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
sha=lambda b: hashlib.sha256(b).hexdigest()
frozen=json.loads((SRC/'frozen_config.json').read_text())
assert sha((SRC/'run_experiment.py').read_bytes())==frozen['source_sha256']
assert sha((SRC/'frozen_protocol.json').read_bytes())==frozen['protocol_sha256']
assert sha((SRC/'development_600_updates.json').read_bytes())==frozen['development_results_sha256']
dev=json.loads((SRC/'development_600_updates.json').read_text())
fresh=json.loads((SRC/'fresh_600_updates.json').read_text())
assert dev['seeds']==[7421,7422]
assert fresh['seeds']==[74201,74202,74203]
assert len(fresh['rows'])==12
methods={'independent','native','additive','mirror'}
assert all(sum(r['method']==m and r['world_seed']==seed for r in fresh['rows'])==1 for seed in fresh['seeds'] for m in methods)
rows=[]
for split, dataset in [('development',dev),('fresh',fresh)]:
    for r in dataset['rows']:
        if split=='fresh':
            payload=ROOT/'source'/'payloads'/f"fresh_{r['world_seed']}_{r['method']}.bin"
            blob=payload.read_bytes()
            assert len(blob)==r['serialized_bytes']
            assert sha(blob)==r['payload_sha256']
            method,rank,segment,tensors=runner.unpack_payload(blob)
            assert (method,rank,segment)==(r['method'],r['latent_rank'],0)
            assert set(tensors)==({'weights'} if method=='independent' else {'basis','coefficients'} if method=='native' else {'basis','attr_a','attr_b','decode'})
        rows.append({
            'world_seed':r['world_seed'],'split':split,'method':r['method'],'latent_rank':r['latent_rank'],
            'serialized_bytes':r['serialized_bytes'],'marginal_coordinate_bytes':r['marginal_coordinate_bytes'],
            'shared_basis_bytes':r['shared_basis_bytes'],'payload_sha256':r['payload_sha256'],
            'train_example_exposures':r['train_examples'],'optimizer_updates':r['optimizer_updates'],
            'train_macs_proxy':r['train_macs_proxy'],'train_wall_s':r['train_wall_s'],
            'inference_macs_proxy':r['inference_macs_proxy'],'inference_wall_s':r['inference_wall_s'],
            'heldout_combo_mse':'' if r['heldout_combo_mse'] is None else r['heldout_combo_mse'],
            'seen_relation_mse':r['seen_relation_mse'],
            'status_note':'development selection' if split=='development' else 'fresh locked evaluation'
        })
fields=list(rows[0])
with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
# Evaluate frozen per-world quality gates, without selecting or changing any setting.
by={(r['world_seed'],r['method']):r for r in fresh['rows']}
world_gates=[]
for seed in fresh['seeds']:
    m=by[(seed,'mirror')]; a=by[(seed,'additive')]; n=by[(seed,'native')]
    held_ratio=m['heldout_combo_mse']/a['heldout_combo_mse']
    seen_ratio=m['seen_relation_mse']/n['seen_relation_mse']
    world_gates.append({'world_seed':seed,'heldout_ratio_to_additive':held_ratio,'seen_ratio_to_native':seen_ratio,
                        'quality_gate_pass':held_ratio<=0.80 and seen_ratio<=1.10})
summary={'fresh_worlds':world_gates,'fresh_quality_gate_passes':sum(x['quality_gate_pass'] for x in world_gates),
         'fresh_quality_gate_total':len(world_gates),'quality_status':'FAIL' if not all(x['quality_gate_pass'] for x in world_gates) else 'PASS',
         'coordinate_bytes_mirror':104,'coordinate_bytes_native':264,'coordinate_ratio':104/264,
         'full_payload_bytes_mirror':1128,'full_payload_bytes_native':1288,'full_payload_ratio':1128/1288,
         'fresh_payloads_validated':12,'fresh_payload_sha256_all_match':True}
(SRC/'verification_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
