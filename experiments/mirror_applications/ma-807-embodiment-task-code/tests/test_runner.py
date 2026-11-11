import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma807',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_heldout_embodiment_task_pairs_and_native_alias():
    rows=m.run();by={r['method']:r for r in rows}
    a=by['mirror_embodiment_task_views'];b=by['native_additive_givens_composition']
    assert a['payload_sha256']==b['payload_sha256']
    assert a['heldout_action_nrmse']<.02 and a['task_success_fraction']>=.95
    assert a['serialized_bytes']<by['independent_pair_policies']['serialized_bytes']
    assert a['serialized_bytes']>.75*by['independent_pair_policies']['serialized_bytes']
    assert a['max_orthogonality_error']<1e-5
    assert max(r['payload_roundtrip_max_abs'] for r in rows)==0
