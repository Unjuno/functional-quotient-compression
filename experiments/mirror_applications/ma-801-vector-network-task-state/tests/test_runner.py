import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma801',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_task_state_holdouts_local_inference_and_native_alias():
    rows=m.run();by={r['method']:r for r in rows}
    a=by['mirror_task_state_views'];b=by['native_cp_coefficient_factorization']
    assert a['payload_sha256']==b['payload_sha256']
    assert a['heldout_output_nrmse']==b['heldout_output_nrmse']<.02
    assert a['average_active_atoms']==2 and a['unique_local_atom_sets']>1
    assert a['serialized_bytes']>.75*by['independent_task_state_coefficients']['serialized_bytes']
    assert max(r['payload_roundtrip_max_abs'] for r in rows)==0
