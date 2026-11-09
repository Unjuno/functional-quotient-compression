import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma789',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_cluster_global_basis_frontier_and_native_alias():
    rows=m.run();by={(r['private_noise_sigma'],r['method']):r for r in rows}
    for sig in (0.0,.05):
        a=by[(sig,'mirror_global_shared_basis')];b=by[(sig,'native_global_pca')]
        assert a['payload_sha256']==b['payload_sha256']
        assert a['heldout_operator_nrmse']==b['heldout_operator_nrmse']
        assert a['router_accuracy']==1.0
    aligned=by[(0.0,'mirror_global_shared_basis')];full=by[(0.0,'independent_full_experts')]
    assert aligned['heldout_operator_nrmse']<.02 and aligned['serialized_bytes']<.75*full['serialized_bytes']
    assert by[(.05,'mirror_global_shared_basis')]['heldout_operator_nrmse']>.02
