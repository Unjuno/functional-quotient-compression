import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma805',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_fourier_regime_view_and_native_control():
    rows=m.run();by={r['method']:r for r in rows}
    a=by['mirror_diffusivity_views'];b=by['native_conditional_fourier_kno'];shared=by['shared_unconditioned_kno'];fno=by['independent_fno_multiplier_tables']
    assert a['payload_sha256']==b['payload_sha256']
    assert a['heldout_field_nrmse']==b['heldout_field_nrmse']<.02
    assert a['serialized_bytes']<.75*fno['serialized_bytes']
    assert shared['heldout_field_nrmse']<.02 and shared['serialized_bytes']<a['serialized_bytes']
    assert a['max_multiplier_magnitude']<=1
    assert max(r['payload_roundtrip_max_abs'] for r in rows)==0
