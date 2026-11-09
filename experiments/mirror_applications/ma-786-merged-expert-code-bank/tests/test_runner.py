import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma786',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_routing_weighted_merge_native_alias_and_private_cost():
    rows=m.run();by={(r['heterogeneity_sigma'],r['method'],r['private_rank']):r for r in rows}
    for sig in (0.0,.1):
        a=by[(sig,'mirror_member_code_bank',0)];b=by[(sig,'native_pca_code_bank',0)]
        assert a['payload_sha256']==b['payload_sha256']
        assert a['heldout_member_nrmse']==b['heldout_member_nrmse']
    aligned=by[(0.0,'mirror_member_code_bank',0)];full=by[(0.0,'independent_members',16)]
    assert aligned['heldout_member_nrmse']<.02 and aligned['serialized_bytes']<.75*full['serialized_bytes']
    h=.1; q8=by[(h,'shared_plus_private_member_residual',8)];q16=by[(h,'shared_plus_private_member_residual',16)];full=by[(h,'independent_members',16)]
    assert q8['heldout_member_nrmse']>.02
    assert q16['heldout_member_nrmse']<1e-5 and q16['serialized_bytes']>full['serialized_bytes']
