import importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma528_run',ROOT/'source'/'run_behavior_codes.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
def test_residual_pool_is_deterministic_unique_and_from_dictionary():
 rng=np.random.default_rng(7);v=rng.normal(size=(16,512)).astype(np.float32);W=rng.normal(size=(2048,512)).astype(np.float32)
 a=r.residual_pool(v,W);b=r.residual_pool(v,W)
 assert np.array_equal(a,b) and len(a)==64 and len(set(a.tolist()))==64
 assert int(a.min())>=0 and int(a.max())<2048
def test_omp_uses_sixteen_unique_local_codes_and_reconstructs():
 rng=np.random.default_rng(8);v=rng.normal(size=(3,512)).astype(np.float32);W=rng.normal(size=(64,512)).astype(np.float32)
 ids,vals,dec=r.omp(v,W,16)
 assert ids.shape==(3,16) and vals.shape==(3,16) and dec.shape==v.shape
 assert all(len(set(row.tolist()))==16 for row in ids)
 expected=np.stack([vals[i]@W[ids[i]] for i in range(3)])
 assert np.allclose(dec,expected,rtol=2e-7,atol=2e-7)
def test_omp_atom_normalization_has_no_broadcast_expansion():
 W=np.zeros((64,512),dtype=np.float32);W[:,0]=1;v=np.zeros((1,512),dtype=np.float32);v[0,0]=2
 ids,vals,dec=r.omp(v,W,16)
 assert dec.shape==(1,512) and np.all(np.isfinite(dec))
 assert np.allclose(dec[0,0],2)
