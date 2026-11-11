import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'source'))
from run_experiment import run

def test_mirror_generator_aliases_native_ntk_generator():
    rows=run();d={r['method']:r for r in rows}
    assert len(rows)==6
    a=d['native_ntk_no_task_storage'];b=d['mirror_generated_coordinate'];c=d['direct_descriptor_conditioning']
    assert a['payload_sha256']==b['payload_sha256']==c['payload_sha256']
    assert a['unseen_descriptor_nrmse']==b['unseen_descriptor_nrmse']==c['unseen_descriptor_nrmse']
    assert b['total_serialized_state_bytes']==a['total_serialized_state_bytes']
    assert d['mirror_cached_task_codes']['total_serialized_state_bytes']==d['mirror_cached_task_codes']['model_payload_bytes']
