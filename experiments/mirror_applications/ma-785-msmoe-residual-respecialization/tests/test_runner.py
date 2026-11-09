import csv,importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma785',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_merge_shared_private_frontier_and_native_alias():
    rows=m.run();by={(r['heterogeneity_sigma'],r['method'],r['private_rank']):r for r in rows}
    for sigma in (0.0,0.05,0.2):
        a=by[(sigma,'mirror_shared_rank2',0)];b=by[(sigma,'native_rank2_decompression',0)]
        assert a['payload_sha256']==b['payload_sha256']
        assert a['heldout_operator_nrmse']==b['heldout_operator_nrmse']
    zero=by[(0.0,'mirror_shared_rank2',0)];full=by[(0.0,'independent_full_bank',12)]
    assert zero['heldout_operator_nrmse']<.02 and zero['serialized_bytes']<.75*full['serialized_bytes']
    q8=by[(0.05,'shared_plus_private_lowrank',8)]
    q12=by[(0.05,'shared_plus_private_lowrank',12)];full=by[(0.05,'independent_full_bank',12)]
    assert q8['heldout_operator_nrmse']>.02
    assert q12['heldout_operator_nrmse']<1e-5 and q12['serialized_bytes']>full['serialized_bytes']
