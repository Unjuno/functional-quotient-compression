import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma784',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_pool_sweep_bytes_quality_and_native_alias():
    rows=m.run();by={r['method']:r for r in rows}
    for k in (1,2):
        a=by[f'mirror_k{k}'];b=by[f'native_givens_k{k}']
        assert a['payload_sha256']==b['payload_sha256']
        assert a['heldout_operator_nrmse']==b['heldout_operator_nrmse']
        assert a['physical_pool_k']==k
    assert by['mirror_k1']['heldout_operator_nrmse']<.02
    assert by['mirror_k1']['serialized_bytes'] < by['independent_full_k4']['serialized_bytes']
    assert by['mirror_k1']['serialized_bytes'] >= .75*by['independent_full_k4']['serialized_bytes']
    assert by['pool_k1_no_view']['heldout_operator_nrmse']>.02
    assert by['pool_k2_no_view']['heldout_operator_nrmse']>.02
