"""Replay MA-882 integrity, actual-byte, and preregistered gate checks."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'fresh_results.json').read_text())
assert r['selected_pca_rank']==256 and r['selected_cp_rank']==8
assert [x['seed'] for x in r['rows']]==[983,997,1009]
rows=[]
# Calibration partitions and seed chunks are disjoint within each ordered pair.
for world in r['rows']:
    for pi in range(2):
        fit=set(world['prompt_rows']['map_fit'][pi]); route=set(world['prompt_rows']['route'][pi]); test=set(world['prompt_rows']['test'][pi])
        assert not (fit & route or fit & test or route & test)
# Cross-pair source conditions may share the same Wikitext row; count that dependence.
cross_pair_overlap={}
for split in ('map_fit','route','test'):
    cross_pair_overlap[split]=sum(len(set(w['prompt_rows'][split][0]) & set(w['prompt_rows'][split][1])) for w in r['rows'])
for pi in range(2):
    seen_fit=set(); seen_route=set(); seen_test=set()
    for world in r['rows']:
        fit=set(world['prompt_rows']['map_fit'][pi]); route=set(world['prompt_rows']['route'][pi]); test=set(world['prompt_rows']['test'][pi])
        assert not (seen_fit & fit or seen_route & route or seen_test & test)
        seen_fit|=fit; seen_route|=route; seen_test|=test
for world in r['rows']:
    seed=world['seed']; rt=world['payload_roundtrip']
    assert rt['seed']==seed and rt['max_abs_mean_nll_diff']<1e-3
    assert world['native_identity_max_abs_nll_diff']<1e-3
    for method in ('full','pca','cp'):
        path=ROOT/'artifacts'/f'{method}_seed_{seed}.bin'; raw=path.read_bytes()
        assert len(raw)==world['payload_bytes'][method]
        assert hashlib.sha256(raw).hexdigest()==world['payload_sha256'][method]
        assert rt['payload_bytes'][method]==len(raw) and rt['payload_sha256'][method]==hashlib.sha256(raw).hexdigest()
    bytes_ok=(world['payload_bytes']['cp']<=.5*world['payload_bytes']['full'] and world['payload_bytes']['cp']<world['payload_bytes']['pca'])
    for pair in world['pair_results']:
        q=(pair['cp_nll']<=pair['full_nll']+.5 and pair['cp_nll']<=pair['pca_nll']+.25 and pair['cp_nll']<=pair['native_nll']+1.)
        rows.append({'seed':seed,'pair':pair['pair'],'quality_pass':q,'storage_pass':bytes_ok,
                     'delta_full':pair['cp_nll']-pair['full_nll'],'delta_pca':pair['cp_nll']-pair['pca_nll'],
                     'delta_native':pair['cp_nll']-pair['native_nll'],'bytes':world['payload_bytes']})
assert all(x['storage_pass'] for x in rows)
assert not any(x['quality_pass'] for x in rows)
print(json.dumps({'experiment_id':'MA-882','payload_and_metric_integrity':'PASS','fresh_pair_seed_conditions':len(rows),
                  'storage_gates_passed':sum(x['storage_pass'] for x in rows),'quality_gates_passed':sum(x['quality_pass'] for x in rows),'cross_pair_prompt_row_overlap_totals':cross_pair_overlap,'gates':rows},indent=2))
