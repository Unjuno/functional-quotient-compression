import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'source'))
from run_experiment import run

def test_private_residual_ranks_and_native_aliases():
    rows=run();assert len(rows)==22
    for scale in (.03,.20):
        d={(r['method'],r['private_rank']):r for r in rows if r['delta_scale']==scale}
        for k in (0,1,2,4):
            a=d[('mirror_private_rank'+str(k),k)];b=d[('native_pca_private_rank'+str(k),k)]
            assert a['serialized_bytes']==b['serialized_bytes']
            assert a['payload_sha256']==b['payload_sha256']
            assert a['heldout_nonlinear_nrmse']==b['heldout_nonlinear_nrmse']
        assert d[('teacher_upper',97)]['heldout_nonlinear_nrmse'] < d[('independent_tangent',97)]['heldout_nonlinear_nrmse']+1e-12
        if scale==.03:
            shared=d[('mirror_private_rank0',0)]
            full_resid=d[('mirror_private_rank4',4)]
            indep=d[('independent_tangent',97)]
            assert shared['heldout_nonlinear_nrmse'] < .02
            assert full_resid['heldout_nonlinear_nrmse'] < shared['heldout_nonlinear_nrmse']
            assert full_resid['serialized_bytes'] > indep['serialized_bytes']
