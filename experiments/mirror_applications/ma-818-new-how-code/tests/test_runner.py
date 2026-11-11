import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma818',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_new_how_misses_threshold_and_aliases_native():
 rows=m.run();by={r['method']:r for r in rows};a=by['mirror_new_how_code'];n=by['native_operator_coordinate'];e=by['explicit_new_operator']
 assert a['heldout_output_nrmse']==0 and n['heldout_output_nrmse']==0
 assert a['serialized_bytes']>.75*e['serialized_bytes']
 assert n['serialized_bytes']<a['serialized_bytes']+10
 assert all(r['payload_roundtrip_max_abs']==0 for r in rows)
