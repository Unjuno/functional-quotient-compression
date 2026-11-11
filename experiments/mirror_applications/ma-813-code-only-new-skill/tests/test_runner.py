import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma813',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_code_alias_and_full_byte_frontier():
 rows=m.run();by={r['method']:r for r in rows};a=by['mirror_new_skill_code'];n=by['native_shared_basis_code']
 assert a['heldout_new_skill_nrmse']<.02
 assert a['serialized_bytes']>by['fresh_rank2_lora']['serialized_bytes']
 assert a['serialized_bytes']>by['independent_full_new_skill']['serialized_bytes']
 assert abs(a['heldout_new_skill_nrmse']-n['heldout_new_skill_nrmse'])<1e-7
 assert all(r['payload_roundtrip_max_abs']==0 for r in rows)
