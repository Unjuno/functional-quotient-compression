import importlib.util
from pathlib import Path
import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/"source/run_experiment.py"
spec=importlib.util.spec_from_file_location("ma527_run",SCRIPT)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_givens_is_orthogonal_and_sparse_projection_bounded():
    G,angles=mod.givens_matrix(.1)
    assert np.allclose(G@G.T,np.eye(32),atol=1e-6)
    assert len(angles)==16
    x=np.zeros((2,32),dtype=np.float32);x[0,:8]=np.arange(1,9);x[1,16:24]=np.arange(1,9)
    y=x@G.T
    ids,vals=mod.sparse_topk(y,8)
    assert ids.shape==(2,8) and vals.shape==(2,8)
    assert np.count_nonzero(vals,axis=1).max()<=8
    assert np.allclose(np.linalg.norm(y,axis=1),np.linalg.norm(x,axis=1),atol=1e-5)

def test_topk_local_feature_codes_roundtrip_decode_shape():
    atoms=np.array([[0.,0.,0.],[0.,0.,0.],[0.,1.,0.],[1.,0.,0.]],dtype=np.float32)
    z=np.array([[1.,0.,-2.,3.]],dtype=np.float32)
    ids,vals=mod.sparse_topk(z,2)
    out=mod.decode_codes(ids.astype(np.int64),vals,atoms)
    assert out.shape==(1,3)
    assert np.allclose(out,np.array([[3.,-2.,0.]],dtype=np.float32))
