import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'source'))
from run_experiment import run

def test_causal_memory_checks_and_native_aliases():
    rows=run();assert len(rows)==8
    d={r['method']:r for r in rows}
    for k in (0,1,3):
        a=d['mirror_fisher_basis_rank'+str(k)];b=d['native_fisher_pca_rank'+str(k)]
        assert a['payload_sha256']==b['payload_sha256']
        assert a['reactivation_nrmse']==b['reactivation_nrmse']
        assert a['sufficiency_composition_nrmse']==b['sufficiency_composition_nrmse']
    full=d['independent_full_traces']
    assert full['reactivation_nrmse']<1e-7
    assert full['specificity_top1']==1.0
    assert full['necessity_erasure_fraction']>.999
    assert full['erase_other_memory_leak']<1e-8
