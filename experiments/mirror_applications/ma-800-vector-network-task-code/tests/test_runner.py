import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma800',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_local_atom_inference_and_factorization_alias():
    rows=m.run();by={r['method']:r for r in rows}
    a=by['mirror_factorized_task_code'];b=by['native_lowrank_coefficient_factorization']
    assert a['payload_sha256']==b['payload_sha256']
    assert a['heldout_output_nrmse']==b['heldout_output_nrmse']<.02
    assert a['average_active_atoms']==2
    assert a['unique_local_atom_sets']>1
    assert a['serialized_bytes']>by['independent_task_coefficients']['serialized_bytes']
    assert max(r['payload_roundtrip_max_abs'] for r in rows)==0
