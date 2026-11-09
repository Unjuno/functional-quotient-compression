import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma788',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_utility_hybrid_beats_random_but_view_is_native_lowrank():
    rows=m.run();by={r['method']:r for r in rows}
    a=by['all_view'];b=by['native_rank1_lora_all_view']
    assert a['payload_sha256']==b['payload_sha256']
    assert a['utility_weighted_nrmse']==b['utility_weighted_nrmse']
    assert by['utility_hybrid_k2']['utility_weighted_nrmse']<.05
    assert by['utility_hybrid_k2']['utility_weighted_nrmse'] < .2*by['random_hybrid_k2']['utility_weighted_nrmse']
    assert by['utility_hybrid_k2']['serialized_bytes'] < .5*by['all_physical_copy']['serialized_bytes']
