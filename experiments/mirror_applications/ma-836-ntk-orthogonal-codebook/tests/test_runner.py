import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'source'))
from run_experiment import run

def test_ntk_control_alias_and_schema():
    rows=run(); d={r['method']:r for r in rows}
    assert len(rows)==7
    assert d['mirror_ntk_codebook']['serialized_bytes']==d['native_ntk_weighted_rank2']['serialized_bytes']
    assert d['mirror_ntk_codebook']['payload_sha256']==d['native_ntk_weighted_rank2']['payload_sha256']
    assert d['mirror_ntk_codebook']['heldout_nrmse']==d['native_ntk_weighted_rank2']['heldout_nrmse']
    assert d['independent_full_updates']['heldout_nrmse'] < d['shared_mean_update']['heldout_nrmse']
    assert d['ntk_orthogonal_basis_fullrank']['basis_orthogonality_error'] < 1e-5
    assert d['ntk_orthogonal_basis_fullrank']['heldout_nrmse'] < 1e-10
