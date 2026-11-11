import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'source'))
from run_experiment import run

def test_fallback_preserves_retention_and_matches_native_control():
    rows=run();assert len(rows)==5;d={r['method']:r for r in rows}
    a=d['mirror_shared_code_private_fallback'];b=d['native_lowrank_private_fallback']
    assert a['payload_sha256']==b['payload_sha256']
    assert a['fallback_count']==8
    assert a['sequential_retrieval_nrmse']<.02
    assert a['specificity_top1']==1.0
    assert d['shared_code_no_fallback']['sequential_retrieval_nrmse']>.02
    assert a['serialized_bytes']<d['independent_full_trace_table']['serialized_bytes']
