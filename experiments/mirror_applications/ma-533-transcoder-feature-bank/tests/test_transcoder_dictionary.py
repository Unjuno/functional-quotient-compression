import importlib.util,tempfile,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('ma533_run',ROOT/'source'/'run_transcoder.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
def test_topk_relu_has_fixed_sparse_support_and_gradients():
 z=torch.tensor([[1.,4.,2.,3.,-1.]],requires_grad=True);s,ids=r.topk_relu(z,2)
 assert torch.count_nonzero(s,dim=1).tolist()==[2]
 assert sorted(ids.tolist()[0]) in ([1,3],[1,2])
 s.sum().backward();assert z.grad is not None and torch.count_nonzero(z.grad)==2
def test_residual_pool_is_unique_and_deterministic():
 rng=np.random.default_rng(33);v=rng.normal(size=(16,512)).astype(np.float32);D=rng.normal(size=(2048,512)).astype(np.float32)
 a=r.residual_pool(v,D);b=r.residual_pool(v,D)
 assert np.array_equal(a,b) and len(set(a.tolist()))==16
def test_omp16_decodes_from_recorded_indices():
 rng=np.random.default_rng(34);v=rng.normal(size=(4,512)).astype(np.float32);D=rng.normal(size=(16,512)).astype(np.float32)
 ids,c,y=r.omp(v,D,16);expected=np.stack([c[i]@D[ids[i]] for i in range(4)])
 assert ids.shape==(4,16) and all(len(set(x.tolist()))==16 for x in ids)
 assert np.allclose(y,expected,rtol=2e-7,atol=2e-7)
def test_dictionary_and_code_payload_roundtrip_is_uncompressed():
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'view.npz';a={'decoder_atoms':np.ones((16,512),np.float32),'pool_ids':np.arange(16,dtype=np.int16),'coefficients':np.ones((16,16),np.float32)};r.save(p,a);b=np.load(p,allow_pickle=False)
  assert all(np.array_equal(a[k],b[k]) for k in a)
  with zipfile.ZipFile(p) as z:assert all(x.compress_type==zipfile.ZIP_STORED for x in z.infolist())
