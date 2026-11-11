import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma804',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_sparse_private_nonlinearity_and_native_alias():
    rows=m.run();by={r['method']:r for r in rows}
    a=by['mirror_shared_operator_sparse_residual'];b=by['native_shared_operator_private_residual'];shared=by['shared_operator_no_private_residual']
    assert a['payload_sha256']==b['payload_sha256']
    assert a['horizon40_nrmse']==b['horizon40_nrmse']<.02
    assert shared['common_regime_nrmse']<.02 and shared['exceptional_regime_nrmse']>.02
    assert a['active_private_residual_count']==2
    assert a['max_local_jacobian_spectral_radius']<1
    assert a['serialized_bytes']<by['independent_full_dynamics']['serialized_bytes']
    assert a['serialized_bytes']>.75*by['independent_full_dynamics']['serialized_bytes']
    assert max(r['payload_roundtrip_max_abs'] for r in rows)==0
