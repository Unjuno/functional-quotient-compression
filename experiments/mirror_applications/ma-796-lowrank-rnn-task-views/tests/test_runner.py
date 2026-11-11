import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma796',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_heldout_recurrent_composition_and_native_alias():
    rows=m.run();by={r['method']:r for r in rows}
    a=by['mirror_recurrent_task_views'];b=by['native_lowrank_task_conditioning']
    assert a['payload_sha256']==b['payload_sha256']
    assert a['heldout_rollout_nrmse']==b['heldout_rollout_nrmse']<.02
    assert a['serialized_bytes']<.75*by['independent_full_task_matrices']['serialized_bytes']
    assert a['max_spectral_radius']<1
    assert a['b1_ablation_mean_nrmse']>.02 and a['b2_ablation_mean_nrmse']>.02
    assert max(r['payload_roundtrip_max_abs'] for r in rows)<1e-5
