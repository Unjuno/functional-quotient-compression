import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma803',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_spectral_view_stability_alias_and_native_frontier():
    rows=m.run();by={r['method']:r for r in rows}
    a=by['mirror_factorized_eigenmode_codes'];b=by['native_spectral_factorization'];diag=by['native_diagonal_eigenvalue_table']
    assert a['payload_sha256']==b['payload_sha256']
    assert a['heldout_horizon48_nrmse']==b['heldout_horizon48_nrmse']<.02
    assert a['max_registered_spectral_radius']<1 and a['out_of_grid_code_spectral_radius']<1
    assert a['serialized_bytes']>.75*by['independent_dense_modal_operators']['serialized_bytes']
    assert diag['serialized_bytes']<a['serialized_bytes']
    assert diag['active_macs_per_step']==a['active_macs_per_step']<by['independent_dense_modal_operators']['active_macs_per_step']
    assert max(r['payload_roundtrip_max_abs'] for r in rows)==0
