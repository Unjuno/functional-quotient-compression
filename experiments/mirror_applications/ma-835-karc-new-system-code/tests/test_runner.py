import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'source'))
from run_experiment import run

def test_native_code_alias_and_result_shape():
    rows=run(); d={r['method']:r for r in rows}
    assert len(rows)==5
    assert d['mirror_system_codes']['heldout_rollout_nrmse']==d['native_rank2_readout_factor']['heldout_rollout_nrmse']
    assert d['mirror_system_codes']['serialized_bytes']==d['native_rank2_readout_factor']['serialized_bytes']
    assert d['basis_growth_cubic_control']['heldout_rollout_nrmse'] < d['shared_only']['heldout_rollout_nrmse']
