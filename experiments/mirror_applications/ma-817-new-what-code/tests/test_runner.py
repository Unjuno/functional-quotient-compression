import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma817',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_native_operand_code_and_full_payload_cost():
 rows=m.run();by={r['method']:r for r in rows};a=by['mirror_new_what_code'];n=by['native_coordinate_code']
 assert a['heldout_output_nrmse']<.02 and a['old_what_retention_nrmse']==0
 assert a['serialized_bytes']>by['explicit_new_what']['serialized_bytes']
 assert abs(a['heldout_output_nrmse']-n['heldout_output_nrmse'])<1e-7
 assert all(r['payload_roundtrip_max_abs']==0 for r in rows)
