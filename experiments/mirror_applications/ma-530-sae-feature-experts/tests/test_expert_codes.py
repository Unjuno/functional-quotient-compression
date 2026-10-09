import importlib.util,tempfile,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('ma530_run',ROOT/'source'/'run_experts.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
def test_router_standardization_and_argmax_are_deterministic():
 mu=np.zeros(512,np.float32);sd=np.ones(512,np.float32);w=np.zeros((16,512),np.float32);b=np.arange(16,dtype=np.float32)
 x=np.zeros((3,512),np.float32);pred,logits=r.route_features(x,(mu,sd,w,b))
 assert pred.tolist()==[15,15,15] and logits.shape==(3,16)
def test_shared_pool_is_deterministic_unique():
 rng=np.random.default_rng(12);v=rng.normal(size=(16,512)).astype(np.float32);W=rng.normal(size=(2048,512)).astype(np.float32)
 a=r.residual_pool(v,W);b=r.residual_pool(v,W)
 assert np.array_equal(a,b) and len(set(a.tolist()))==64
def test_sparse_expert_codes_decode_exactly_within_fp32_tolerance():
 rng=np.random.default_rng(13);v=rng.normal(size=(16,512)).astype(np.float32);W=rng.normal(size=(64,512)).astype(np.float32)
 ids,c,dec=r.omp(v,W,16);expected=np.stack([c[i]@W[ids[i]] for i in range(16)])
 assert all(len(set(row.tolist()))==16 for row in ids)
 assert np.allclose(dec,expected,rtol=2e-7,atol=2e-7)
def test_uncompressed_paid_payload_roundtrip():
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'bank.npz';arrays={'router':np.ones((16,512),np.float32),'pool':np.arange(64,dtype=np.int16)};r.save(p,arrays);z=np.load(p,allow_pickle=False)
  assert all(np.array_equal(arrays[k],z[k]) for k in arrays)
  with zipfile.ZipFile(p) as f:assert all(x.compress_type==zipfile.ZIP_STORED for x in f.infolist())
