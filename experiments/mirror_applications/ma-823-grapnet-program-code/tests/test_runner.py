import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma823',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_native_sparse_program_alias_and_fixed_topology_fail():
 rows=m.run();by={r['method']:r for r in rows};a=by['mirror_graph_program_code'];n=by['native_sparse_edge_indices'];fixed=by['shared_fixed_topology'];dense=by['independent_dense_graphs']
 assert a['heldout_program_nrmse']==n['heldout_program_nrmse']==0
 assert a['serialized_bytes']==n['serialized_bytes']
 assert fixed['heldout_program_nrmse']>.5
 assert dense['serialized_bytes']<a['serialized_bytes']
 assert all(r['payload_roundtrip_max_abs']==0 for r in rows)
