import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'source'))
from run_experiment import run

def test_native_pca_alias_and_two_scales():
    rows=run();
    assert len(rows)==10
    for scale in (.03,.20):
        d={r['method']:r for r in rows if r['delta_scale']==scale}
        assert d['native_pca_rank4']['serialized_bytes']==d['mirror_rank4']['serialized_bytes']
        assert d['native_pca_rank4']['payload_sha256']==d['mirror_rank4']['payload_sha256']
        assert d['native_pca_rank4']['heldout_nonlinear_nrmse']==d['mirror_rank4']['heldout_nonlinear_nrmse']
        assert d['teacher_upper']['heldout_nonlinear_nrmse'] < d['full_tangent']['heldout_nonlinear_nrmse']
