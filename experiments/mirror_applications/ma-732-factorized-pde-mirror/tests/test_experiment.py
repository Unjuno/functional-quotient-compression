import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_experiment import evaluate

def test_direct_conditioned_and_mirror_match():
    rows,meta=evaluate(73201)
    assert meta['mirror_direct_equal_bytes']
    for r in rows:
        if r['method']=='factorized_mirror':
            mate=next(x for x in rows if x['world_or_seed']==r['world_or_seed'] and x['status_note']==r['status_note'] and x['method']=='direct_conditioned_deeponet')
            assert abs(float(r['primary_value'])-float(mate['primary_value']))<1e-14

def test_actual_map_bank_is_complete():
    _,meta=evaluate(73211)
    assert meta['heldout_combinations']==4
    assert meta['independent_map_bytes']>0
    assert meta['mirror_bytes']>0
