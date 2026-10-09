import importlib.util
from pathlib import Path
import numpy as np

SRC=Path(__file__).resolve().parents[1]/'source'/'run_distillation.py'
spec=importlib.util.spec_from_file_location('ma520_run_distillation',SRC)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_quantized_control_shapes_and_error():
    x=np.random.default_rng(5).normal(size=(16,512)).astype(np.float32)
    q,s,y=mod.quantize_per_vector(x)
    assert q.shape==(16,512) and q.dtype==np.int8
    assert s.shape==(16,) and s.dtype==np.float32
    assert y.shape==x.shape and np.max(np.abs(x-y))<0.02

def test_native_pca_has_fixed_shared_rank_and_heldout_codes():
    rng=np.random.default_rng(8); x=rng.normal(size=(16,512)).astype(np.float32)
    mean,basis,codes=mod.native_pca(x)
    assert mean.shape==(512,) and basis.shape==(512,4) and codes.shape==(16,4)
    assert np.isfinite(mean).all() and np.isfinite(basis).all() and np.isfinite(codes).all()

def test_vector_metrics_exact_reconstruction():
    x=np.random.default_rng(2).normal(size=(16,512)).astype(np.float32)
    m=mod.vector_metrics(x,x)
    assert m['heldout_relative_rmse']==0.0
    assert m['heldout_cosine_mean']>0.999999

def test_payload_is_actual_uncompressed_npz(tmp_path):
    import zipfile
    x=np.zeros((16,512),dtype=np.float32)
    n=mod.payload(tmp_path/'v.npz','explicit_fp32',x)
    assert n==(tmp_path/'v.npz').stat().st_size
    with zipfile.ZipFile(tmp_path/'v.npz') as z:
        assert z.namelist()
        assert all(i.compress_type==zipfile.ZIP_STORED for i in z.infolist())
