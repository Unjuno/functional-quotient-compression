import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma833',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_native_karc_readout_alias_and_shared_only_quality_boundary():
 rows=m.run();by={r['method']:r for r in rows};a=by['mirror_system_readout_views'];n=by['native_lowrank_readout'];shared=by['shared_karc_readout_only'];ind=by['independent_karc_readouts']
 assert a['heldout_system_rollout_nrmse']<1e-6 and n['heldout_system_rollout_nrmse']<1e-6
 assert abs(a['serialized_bytes']-n['serialized_bytes'])<=8
 assert a['active_feature_evaluations']==n['active_feature_evaluations']
 assert shared['heldout_system_rollout_nrmse']>.02
 assert a['serialized_bytes']<ind['serialized_bytes']
 assert a['rollout_finite'] and n['rollout_finite']
