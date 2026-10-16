import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'source'))
import core
try: import followup
except ImportError: followup=None

def test_fisher_residual_equivalence():
    assert followup is not None, 'followup not implemented'
    w=core.make_world('partial',970,64,32,64)
    p=core.initialize('mirror_r2',8,17,1)
    scores=followup.scores(p,w)
    assert np.max(np.abs(scores['fisher_norm']-scores['train_residual']))<1e-12
    assert np.all(scores['gradient_conflict']>=0)
    w['Ytest'][:]=1e9;w['private_truth']=[0,1]
    again=followup.scores(p,w)
    assert all(np.array_equal(v,again[k]) for k,v in scores.items())

def test_equal_private_bytes_and_selection():
    assert followup is not None, 'followup not implemented'
    s=np.array([0,6,3,7,1,8,2,4.])
    assert followup.top2(s)==[5,3]
