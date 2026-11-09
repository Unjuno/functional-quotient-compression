import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma783',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_mirror_alias_and_frontier_costs():
    rows=m.run(); by={r['method']:r for r in rows}
    a=by['mirror_layer_givens_views']; b=by['native_direct_angle_givens']
    assert a['payload_sha256']==b['payload_sha256']
    assert a['serialized_bytes']==b['serialized_bytes']
    assert a['routed_output_nrmse']==b['routed_output_nrmse']
    assert a['router_plus_expert_macs_per_token_layer']==b['router_plus_expert_macs_per_token_layer']
    assert a['serialized_bytes'] < by['independent_layer_experts']['serialized_bytes']
    assert a['router_plus_expert_macs_per_token_layer'] > by['independent_layer_experts']['router_plus_expert_macs_per_token_layer']
    assert by['unipool_global_experts']['routed_output_nrmse'] > .02
