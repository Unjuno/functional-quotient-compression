import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma827',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_direct_native_coefficients_dominate_endpoint_factor_bytes():
 rows=m.run();by={r['method']:r for r in rows};a=by['mirror_source_target_codes'];n=by['native_edge_coefficients'];so=by['source_only_codes']
 assert a['heldout_edge_nrmse']==0 and n['heldout_edge_nrmse']==0
 assert a['serialized_bytes']>n['serialized_bytes']
 assert so['heldout_edge_nrmse']>.02
 assert a['parent_evaluations']==n['parent_evaluations']
