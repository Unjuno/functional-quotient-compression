import importlib.util
from pathlib import Path
import numpy as np

SRC=Path(__file__).parents[1]/'source'/'run_experiment.py'
spec=importlib.util.spec_from_file_location('ma534_run',SRC)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_givens_preserves_pair_norm():
    x=np.arange(16,dtype=np.float32)[None,:]
    y=m.apply_view(x,np.linspace(-1,1,8,dtype=np.float32),'givens')
    np.testing.assert_allclose(np.sum(x*x,axis=-1),np.sum(y*y,axis=-1),rtol=1e-6,atol=1e-5)

def test_equal_size_pairwise_and_feature_gates():
    x=np.ones((2,16),np.float32)
    pair=m.apply_view(x,np.full(8,2,np.float32),'pairwise')
    feat=m.apply_view(x,np.full(16,2,np.float32),'elementwise')
    np.testing.assert_array_equal(pair,feat)

def test_sparse_encode_selected_slice():
    x=np.ones((3,512),np.float32);E=np.ones((16,512),np.float32);b=np.zeros(16,np.float32)
    z=m.sparse_encode(x,E,b)
    assert z.shape==(3,16)
    assert np.all(np.count_nonzero(z,axis=1)==16)
