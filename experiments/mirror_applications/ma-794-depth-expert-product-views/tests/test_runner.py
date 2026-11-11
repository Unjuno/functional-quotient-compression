import importlib.util
from pathlib import Path

P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma794',P); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)

def test_native_bilinear_alias_and_screen_gates():
    rows=m.run(); by={r['method']:r for r in rows}
    a=by['mirror_product_codes']; b=by['native_bilinear_shared_basis']
    assert a['payload_sha256']==b['payload_sha256']
    assert a['heldout_operator_nrmse']==b['heldout_operator_nrmse']
    assert a['all_role_output_nrmse']==b['all_role_output_nrmse']
    assert a['heldout_operator_nrmse']>.02
    assert a['serialized_bytes']<by['independent_roles']['serialized_bytes']
    assert a['per_layer_load_min']==a['per_layer_load_max']==1
    assert max(r['payload_roundtrip_max_abs'] for r in rows)<1e-6
