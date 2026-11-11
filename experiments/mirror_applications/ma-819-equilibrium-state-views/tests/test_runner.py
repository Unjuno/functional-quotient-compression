import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma819',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_native_system_embedding_alias_and_stability():
 rows=m.run();by={r['method']:r for r in rows};a=by['mirror_system_role_view'];n=by['native_system_embedding']
 assert a['heldout_system_rollout_nrmse']==n['heldout_system_rollout_nrmse']==0
 assert a['serialized_bytes']==n['serialized_bytes']
 assert abs(a['spectral_radius_max']-n['spectral_radius_max'])<1e-7
 assert all(r['payload_roundtrip_max_abs']==0 for r in rows)
 assert max(r['spectral_radius_max'] for r in rows)<1
