import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma824',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_composition_matches_native_and_independent_payload_is_smaller():
 rows=m.run();by={r['method']:r for r in rows};a=by['mirror_factorized_views'];n=by['native_modular_routing'];ind=by['independent_pair_networks'];fixed=by['fixed_topology_function']
 assert a['heldout_pair_output_nrmse']==n['heldout_pair_output_nrmse']==ind['heldout_pair_output_nrmse']==0
 assert ind['serialized_bytes']<a['serialized_bytes']
 assert fixed['heldout_pair_output_nrmse']>.02
 assert a['active_edge_ops']==n['active_edge_ops']
