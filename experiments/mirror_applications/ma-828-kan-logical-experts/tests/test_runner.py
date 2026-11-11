import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma828',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_native_gskan_alias_and_dense_control_limitation():
 rows=m.run();by={r['method']:r for r in rows};a=by['mirror_kan_expert_views'];n=by['native_gskan_experts'];dense=by['dense_moe_experts']
 assert a['heldout_expert_nrmse']==n['heldout_expert_nrmse']==0
 assert a['serialized_bytes']==n['serialized_bytes']
 assert a['active_edge_evaluations']==n['active_edge_evaluations']
 assert dense['heldout_expert_nrmse']>.9
 assert all(r['payload_roundtrip_max_abs']==0 for r in rows if r['method']!='dense_moe_experts')
