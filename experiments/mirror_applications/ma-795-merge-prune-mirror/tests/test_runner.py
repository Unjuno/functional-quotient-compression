import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma795',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_merge_prune_controls_and_native_alias():
    rows=m.run(); by={r['method']:r for r in rows}
    a=by['merge_plus_rank1_view']; b=by['merge_plus_native_rank1']
    assert a['payload_sha256']==b['payload_sha256']
    assert a['weighted_test_output_nrmse']==b['weighted_test_output_nrmse']
    assert max(r['payload_roundtrip_max_abs'] for r in rows)<1e-5
    assert a['serialized_bytes']<.75*by['independent_overcomplete']['serialized_bytes']
    assert a['retained_expert_count']==2
    assert by['merge_only']['retained_expert_count']==2
