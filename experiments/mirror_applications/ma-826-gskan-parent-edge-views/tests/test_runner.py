import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma826',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_gskan_native_alias_and_modulation_cost():
 rows=m.run();by={r['method']:r for r in rows};a=by['mirror_affine_edge_views'];n=by['native_gskan_affine_parent'];mod=by['mirror_affine_modulation_views']
 assert a['heldout_edge_nrmse']==n['heldout_edge_nrmse']==0
 assert a['serialized_bytes']==n['serialized_bytes']
 assert a['parent_function_evaluations']==n['parent_function_evaluations']
 assert mod['serialized_bytes']>a['serialized_bytes']
 assert by['independent_kan_edge_samples']['serialized_bytes']>a['serialized_bytes']
