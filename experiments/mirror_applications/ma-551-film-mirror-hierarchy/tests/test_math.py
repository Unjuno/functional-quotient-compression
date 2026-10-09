import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
spec=importlib.util.spec_from_file_location('ma551',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_rotation_is_orthogonal():
    r=m.rotation(np.linspace(-.5,.5,m.D//2))
    np.testing.assert_allclose(r.T@r,np.eye(m.D),atol=1e-12)

def test_world_generates_shared_film_and_eight_functions():
    a=m.make_world(1)
    assert a[0].shape==(m.D,) and a[2].shape==(m.N_FUNCS,m.D//2)
    assert a[3].shape==(m.N_FUNCS,m.HELD,m.D)
    assert a[4].shape==a[3].shape

def test_rotation_composes_only_disjoint_pairs():
    r=m.rotation(np.zeros(m.D//2));np.testing.assert_array_equal(r,np.eye(m.D))
