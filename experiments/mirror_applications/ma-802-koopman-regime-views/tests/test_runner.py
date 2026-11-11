import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma802',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_stable_heldout_koopman_view_and_native_alias():
    rows=m.run();by={r['method']:r for r in rows}
    a=by['mirror_latent_regime_views'];b=by['native_lowrank_conditional_dynamics']
    assert a['payload_sha256']==b['payload_sha256']
    assert a['heldout_horizon40_nrmse']==b['heldout_horizon40_nrmse']<.02
    assert a['max_spectral_radius']<1
    assert a['serialized_bytes']<.75*by['independent_regime_operators']['serialized_bytes']
    assert max(r['payload_roundtrip_max_abs'] for r in rows)==0
