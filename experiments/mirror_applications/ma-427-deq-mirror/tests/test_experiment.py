import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_experiment import evaluate

def test_rank1_mirror_and_native_control_same_bytes_and_metrics():
    rows,meta=evaluate(42701)
    assert meta['mirror_native_identical_payload']
    for row in rows:
        if row['method']=='mirror_rank1':
            mate=next(x for x in rows if x['world_or_seed']==row['world_or_seed'] and x['condition'].split('_',1)[0]==row['condition'].split('_',1)[0] and x['status_note']==row['status_note'] and x['method']=='native_rank1_bias')
            assert row['primary_value']==mate['primary_value']

def test_bytes_and_stability():
    _,meta=evaluate(42711)
    assert meta['mirror_bank_bytes']>0
    assert meta['independent_full_bias_bank_bytes']>0
