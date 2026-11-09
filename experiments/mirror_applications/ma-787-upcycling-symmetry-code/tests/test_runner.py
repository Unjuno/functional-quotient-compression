import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma787',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_upcycling_code_vs_native_lora_cpt_and_noise():
    rows=m.run();by={(r['heterogeneity_sigma'],r['method']):r for r in rows}
    for sig in (0.0,.05):
        a=by[(sig,'mirror_shared_basis_code')];b=by[(sig,'native_rank1_lora')]
        assert a['payload_sha256']==b['payload_sha256']
        assert a['heldout_child_nrmse']==b['heldout_child_nrmse']
    assert by[(0.0,'mirror_shared_basis_code')]['heldout_child_nrmse']<.02
    assert by[(0.0,'random_perturbation_code')]['heldout_child_nrmse']>.02
    assert by[(0.0,'continued_pretraining_full_children')]['optimizer_updates']==3200
    assert by[(0.0,'continued_pretraining_full_children')]['heldout_child_nrmse']<.02
    assert by[(.05,'mirror_shared_basis_code')]['heldout_child_nrmse']>.02
