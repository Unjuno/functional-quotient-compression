import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma834',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_native_family_selection_matches_view_and_beats_fixed_family():
 rows=m.run();by={r['method']:r for r in rows};a=by['mirror_sparse_basis_views'];n=by['native_karc_family_selection'];full=by['full_heterogeneous_karc_bank'];fixed=by['fixed_fourier_only']
 assert a['selection_accuracy']==n['selection_accuracy']==1
 assert abs(a['heldout_function_nrmse']-n['heldout_function_nrmse'])<1e-8
 assert n['serialized_bytes']<a['serialized_bytes']
 assert a['heldout_function_nrmse']<.02 and fixed['heldout_function_nrmse']>.9
 assert a['active_feature_evaluations']==n['active_feature_evaluations']<full['active_feature_evaluations']+1
