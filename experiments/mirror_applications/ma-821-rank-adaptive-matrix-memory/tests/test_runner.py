import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma821',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_native_svd_matches_rank_adaptive_view():
 rows=m.run();by={r['method']:r for r in rows};a=by['mirror_rank_adaptive'];n=by['native_svd_rank_truncation']
 assert a['serialized_bytes']<.75*by['independent_full']['serialized_bytes']
 assert a['heldout_memory_nrmse']<1e-6 and a['mean_pair_composition_relative_error']<1e-6
 assert abs(a['serialized_bytes']-n['serialized_bytes'])<=2
 assert a['active_macs_per_read']==n['active_macs_per_read']
 assert all(r['payload_roundtrip_max_abs']==0 for r in rows)
