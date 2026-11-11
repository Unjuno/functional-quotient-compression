import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma797',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_locus_views_recover_heldout_and_alias_native_lowrank():
    rows=m.run();by={r['method']:r for r in rows}
    a=by['mirror_factorized_locus_views'];b=by['native_lowrank_locus_conditioning']
    assert a['payload_sha256']==b['payload_sha256']
    assert a['heldout_output_nrmse']==b['heldout_output_nrmse']<.02
    assert a['serialized_bytes']<.75*by['independent_full_task_stages']['serialized_bytes']
    assert max(r['payload_roundtrip_max_abs'] for r in rows)<1e-5
    assert a['correct_locus_view_nrmse']<.02
    assert a['wrong_locus_view_nrmse']>a['correct_locus_view_nrmse']+.01
