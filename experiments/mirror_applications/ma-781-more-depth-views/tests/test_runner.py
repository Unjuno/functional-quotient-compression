import csv, importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma781',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_reused_expert_view_screen_and_native_alias():
    rows=m.run()
    mirror=next(r for r in rows if r['method']=='mirror_depth_expert_views')
    native=next(r for r in rows if r['method']=='native_bilinear_shared_basis')
    assert mirror['payload_sha256']==native['payload_sha256']
    assert mirror['serialized_bytes']==native['serialized_bytes']
    assert mirror['heldout_operator_nrmse']==native['heldout_operator_nrmse']
    assert mirror['active_macs_per_role']==native['active_macs_per_role']
    assert mirror['heldout_operator_nrmse']<=0.02
    independent=next(r for r in rows if r['method']=='independent_layer_expert')
    assert mirror['serialized_bytes'] < independent['serialized_bytes']
