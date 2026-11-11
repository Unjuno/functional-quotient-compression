import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma811',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_native_additive_control_dominates_mirror():
 rows=m.run();by={r['method']:r for r in rows};a=by['mirror_option_position_views'];b=by['native_additive_factors']
 assert a['heldout_sequence_nrmse']==0 and b['heldout_sequence_nrmse']==0
 assert a['serialized_bytes']<.75*by['independent_pair_bank']['serialized_bytes']
 assert b['serialized_bytes']<a['serialized_bytes']
 assert a['active_macs_per_action']==b['active_macs_per_action']
 assert all(r['payload_roundtrip_max_abs']==0 for r in rows)
