import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_fresh_direct_control_equivalence_and_byte_order():
    rows=json.loads((ROOT/'source/fresh_metrics.json').read_text())
    for seed in (29611,29612,29613):
        for condition in ('aligned_orthogonal_orbit','independent_isotropic_deltas'):
            group={r['method']:r for r in rows if r['world_or_seed']==seed and r['condition']==condition}
            mirror=group['mirror_superposition']; direct=group['direct_orbit']
            assert mirror['task_output_norm_mse']==direct['task_output_norm_mse']
            assert mirror['signed_pair_norm_mse']==direct['signed_pair_norm_mse']
            assert mirror['serialized_bytes']>direct['serialized_bytes']
    for r in rows:
        if r['condition']=='aligned_orthogonal_orbit' and r['method']=='mirror_superposition':
            assert r['task_output_norm_mse']<1e-10
