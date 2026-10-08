import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_experiment import evaluate

def test_mirror_matches_mole_control():
    rows,meta=evaluate(67401)
    assert meta['mirror_mole_identical_bytes']
    for r in rows:
        if r['method']=='mirror_top2':
            q=next(x for x in rows if x['status_note']==r['status_note'] and x['method']=='mole_top2')
            assert r['primary_value']==q['primary_value']

def test_sparse_reconstruction_and_private_cost():
    _,m=evaluate(67411)
    assert m['independent_bytes']>0 and m['private_residual_bytes']>0
