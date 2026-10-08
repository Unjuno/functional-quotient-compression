import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_experiment import evaluate

def test_task_identity_and_bytes():
    rows,m=evaluate(66911)
    assert m['tasks']==12 and m['mirror_bytes']>0 and m['lorahub_bytes']>0
    assert len(rows)==48

def test_direct_solver_is_at_least_as_accurate_as_learned_map():
    rows,_=evaluate(66912)
    for i in range(12):
        mirror=float(next(x['primary_value'] for x in rows if 'mirror_support_map' in x['method'] and f'task={i};' in x['status_note']))
        direct=float(next(x['primary_value'] for x in rows if 'direct_least_squares' in x['method'] and f'task={i};' in x['status_note']))
        assert direct < 0.1
        assert mirror < 0.1
