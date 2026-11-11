import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma822',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_native_cyclic_code_dominates_mirror():
 rows=m.run();by={r['method']:r for r in rows};a=by['mirror_group_element_code'];n=by['native_cyclic_group_index'];ind=by['independent_matrix_table']
 assert a['heldout_composition_nrmse']<1e-6 and n['heldout_composition_nrmse']<1e-6
 assert n['serialized_bytes']<a['serialized_bytes']
 assert ind['serialized_bytes']<a['serialized_bytes']
 assert a['max_orthogonality_error']<1e-5 and a['max_determinant_error']<1e-5
 assert all(r['payload_roundtrip_max_abs']<1e-5 for r in rows)
