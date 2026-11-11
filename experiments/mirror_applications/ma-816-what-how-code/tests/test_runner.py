import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma816',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_shared_baseline_and_native_factor_control():
 rows=m.run();by={r['method']:r for r in rows};a=by['mirror_what_how_views'];n=by['native_tensor_factorization'];base=by['shared_core_only']
 assert base['heldout_cross_product_nrmse']<.02
 assert base['serialized_bytes']<a['serialized_bytes']
 assert a['heldout_cross_product_nrmse']==n['heldout_cross_product_nrmse']==0
 assert abs(a['serialized_bytes']-n['serialized_bytes'])<=4
 assert all(r['payload_roundtrip_max_abs']==0 for r in rows)
