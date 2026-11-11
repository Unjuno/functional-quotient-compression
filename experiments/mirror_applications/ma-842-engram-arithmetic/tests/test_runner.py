import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'source'))
from run_experiment import run

def test_code_composition_matches_native_factorization():
    rows=run();assert len(rows)==8;d={r['method']:r for r in rows}
    for k in (0,1,3):
        a=d['mirror_code_superposition_rank'+str(k)];b=d['native_pca_code_addition_rank'+str(k)]
        assert a['payload_sha256']==b['payload_sha256']
        assert a['mixture_composition_nrmse']==b['mixture_composition_nrmse']
    full=d['mirror_code_superposition_rank3'];ind=d['independent_raw_addition']
    assert full['mixture_composition_nrmse']<1e-7
    assert full['necessity_erasure_fraction']>.999
    assert .75*ind['serialized_bytes'] < full['serialized_bytes'] < ind['serialized_bytes']
