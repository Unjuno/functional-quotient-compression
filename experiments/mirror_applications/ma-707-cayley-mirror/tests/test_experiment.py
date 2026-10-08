import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run_experiment import cayley, skew_from, PAIRS, make_base, generators, mirror_matrix, payload_bytes, skew_from

def test_cayley_orthogonal():
    c=np.array([.1,-.2,.03,.04,-.05,.02]); q=cayley(skew_from(c))
    assert np.max(np.abs(q.T@q-np.eye(4)))<1e-12

def test_base_and_views_stable():
    b=make_base(7); assert np.max(np.abs(b.T@b-np.eye(4)))<1e-12
    for c in (np.array([0.,0.]),np.array([.5,-.7])):
        q=mirror_matrix(b,c,[skew_from(np.eye(6)[0]*.5),skew_from(np.eye(6)[1]*.5)])
        assert np.max(np.abs(q.T@q-np.eye(4)))<1e-12
        assert abs(np.max(np.abs(np.linalg.eigvals(q)))-1)<1e-12

def test_serialized_package_deterministic():
    a={'x':np.arange(8,dtype=np.float32)}
    assert payload_bytes(a)==payload_bytes(a)
