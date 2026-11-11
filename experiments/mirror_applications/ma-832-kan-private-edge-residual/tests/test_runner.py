import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma832',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_private_residual_frontier_and_native_control():
 rows=m.run();by={r['method']:r for r in rows};a=by['mirror_sparse_private_residual'];n=by['native_gskan_plus_rational_residual'];shared=by['shared_parent_only'];ind=by['independent_edge_tables']
 assert shared['heldout_edge_nrmse']>.02
 assert a['heldout_edge_nrmse']==n['heldout_edge_nrmse']==0
 assert a['serialized_bytes']<.75*ind['serialized_bytes']
 assert abs(a['serialized_bytes']-n['serialized_bytes'])<=10
 assert a['private_function_evaluations']==n['private_function_evaluations']
