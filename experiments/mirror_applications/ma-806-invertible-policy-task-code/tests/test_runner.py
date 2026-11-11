import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma806',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_one_step_invertible_policy_and_native_givens_alias():
    rows=m.run();by={r['method']:r for r in rows}
    a=by['mirror_one_step_skill_views'];b=by['native_givens_one_step_adapter']
    assert a['payload_sha256']==b['payload_sha256']
    assert a['one_step_action_nrmse']<.02 and a['task_success_fraction']>=.95
    assert a['serialized_bytes']<.75*by['independent_invertible_policies']['serialized_bytes']
    assert a['max_orthogonality_error']<1e-5
    assert a['flow_evaluations_per_action']==1 and a['iterative_flow_evaluations']==4
    assert a['iterative4_action_nrmse']<.02
    assert max(r['payload_roundtrip_max_abs'] for r in rows)==0
