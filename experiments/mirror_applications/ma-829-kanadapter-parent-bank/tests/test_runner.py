import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma829',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_shared_parent_gain_is_native_control_alias():
 rows=m.run();by={r['method']:r for r in rows};a=by['mirror_task_views'];n=by['native_shared_rational_coefficients'];ind=by['independent_kanadapter_bank'];fixed=by['shared_parent_no_task_code']
 assert a['heldout_task_nrmse']==n['heldout_task_nrmse']==0
 assert a['serialized_bytes']<.75*ind['serialized_bytes']
 assert abs(a['serialized_bytes']-n['serialized_bytes'])<=8
 assert fixed['heldout_task_nrmse']>.5
 assert a['active_parent_evaluations']==n['active_parent_evaluations']
