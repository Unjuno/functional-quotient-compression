import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma810',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_composition_and_native_lowrank_alias():
 rows=m.run();by={r['method']:r for r in rows};a=by['mirror_option_codes'];b=by['native_rank2_adapter_basis']
 assert a['heldout_sequence_nrmse']<.02 and b['heldout_sequence_nrmse']<.02
 assert a['serialized_bytes']<.75*by['independent_full_adapter_bank']['serialized_bytes']
 assert a['payload_roundtrip_max_abs']==b['payload_roundtrip_max_abs']==0
 assert a['heldout_sequence_nrmse']==b['heldout_sequence_nrmse']
 assert a['active_macs_per_action']==b['active_macs_per_action']
