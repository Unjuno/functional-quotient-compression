import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma830',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_native_factor_control_is_smaller_and_exact():
 rows=m.run();by={r['method']:r for r in rows};a=by['mirror_task_layer_views'];n=by['native_hypernetwork_factors'];ind=by['independent_pair_functions']
 assert a['heldout_pair_nrmse']==n['heldout_pair_nrmse']==0
 assert n['serialized_bytes']<a['serialized_bytes']
 assert a['parent_evaluations']==n['parent_evaluations']
 assert ind['heldout_pair_nrmse']<.02
 assert by['task_only']['heldout_pair_nrmse']>.02
 assert by['layer_only']['heldout_pair_nrmse']>.02
