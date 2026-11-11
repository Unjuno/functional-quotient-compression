import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma809',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_private_residual_frontier_and_native_control():
 rows=m.run(); by={r['method']:r for r in rows}
 assert by['independent_full']['serialized_bytes']==1075
 assert by['shared_only']['worst_skill_nrmse']>.02
 a=by['mirror_views_plus_private_rank1']; b=by['native_shared_plus_private_rank1']
 assert a['worst_skill_nrmse']<1e-5 and a['success_fraction']==1.0
 assert a['serialized_bytes']>by['independent_full']['serialized_bytes']
 assert abs(a['ood_action_nrmse']-b['ood_action_nrmse'])<1e-7
 assert abs(a['serialized_bytes']-b['serialized_bytes'])<=2
 assert all(r['payload_roundtrip_max_abs']==0 for r in rows)
