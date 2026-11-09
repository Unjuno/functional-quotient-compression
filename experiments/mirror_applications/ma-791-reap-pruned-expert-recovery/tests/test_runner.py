import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma791',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_pruned_recovery_and_private_need():
    rows=m.run();by={(r['heterogeneity_sigma'],r['method']):r for r in rows}
    for sig in (0.0,.05):
        a=by[(sig,'mirror_pruned_expert_views')];b=by[(sig,'native_pca_residual_codes')]
        assert a['payload_sha256']==b['payload_sha256']
        assert a['heldout_pruned_nrmse']==b['heldout_pruned_nrmse']
    aligned=by[(0.0,'mirror_pruned_expert_views')];full=by[(0.0,'independent_full_bank')]
    assert aligned['heldout_pruned_nrmse']<.02 and aligned['serialized_bytes']<.75*full['serialized_bytes']
    assert by[(.05,'mirror_pruned_expert_views')]['heldout_pruned_nrmse']>.02
    assert by[(.05,'shared_plus_private_rank4')]['serialized_bytes']>by[(.05,'independent_full_bank')]['serialized_bytes']
