import importlib.util
from pathlib import Path
import numpy as np

P=Path(__file__).parents[1]/'source'/'run_screen.py'
spec=importlib.util.spec_from_file_location('screen',P); screen=importlib.util.module_from_spec(spec); spec.loader.exec_module(screen)

def test_normalize_and_basis_are_finite():
    x=np.array([[3.,4.],[0.,0.]],np.float32)
    y=screen.normalize(x)
    assert np.all(np.isfinite(y))
    assert np.allclose(y[0],[.6,.8])

def test_fixed_world_generation_replays():
    a=screen.make_world(51001); b=screen.make_world(51001)
    for key in ('keys','effects','mapping','support_x','cal_pos','cal_neg','eval_pos','eval_neg'):
        assert np.array_equal(a[key],b[key])

def test_pca_reconstruction_on_full_rank():
    x=np.eye(4,dtype=np.float32)
    basis=screen.svd_basis(x,4)
    assert np.allclose((x@basis)@basis.T,x,atol=1e-6)
