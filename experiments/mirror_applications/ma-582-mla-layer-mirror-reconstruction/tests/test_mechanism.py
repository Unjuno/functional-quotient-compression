import importlib.util
from pathlib import Path
import numpy as np

SRC=Path(__file__).parents[1]/'source'/'run_experiment.py'
spec=importlib.util.spec_from_file_location('ma582',SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_group_partition_covers_each_layer_once():
    assert sorted(sum((list(g) for g in m.GROUPS),[])) == list(range(m.L))

def test_pca_roundtrip_reconstruction_is_finite():
    rng=np.random.default_rng(582)
    x=rng.normal(size=(48,12)).astype(np.float32)
    mean,basis=m.pca(x,4)
    restored=((x-mean)@basis)@basis.T+mean
    assert basis.shape==(12,4)
    assert np.isfinite(restored).all()
    assert np.mean((x-restored)**2) < np.mean(x*x)

def test_group_view_and_native_control_are_same_shared_basis_path():
    rng=np.random.default_rng(583)
    x=rng.normal(size=(m.PFX,m.F)).astype(np.float32)
    mean,basis=m.pca(x,8)
    residual=x-(((x-mean)@basis)@basis.T+mean)
    role_basis=np.stack([m.pca(residual.reshape(-1,m.ROLE,m.D)[:,r],2)[1] for r in range(m.ROLE)])
    def view_decode():
        z=((x-mean)@basis).astype(np.float16).astype(np.float32)
        base=z@basis.T+mean
        res=(x-base).reshape(-1,m.ROLE,m.D)
        code=np.stack([res[:,r]@role_basis[r] for r in range(m.ROLE)],axis=1).astype(np.float16).astype(np.float32)
        return base+np.stack([code[:,r]@role_basis[r].T for r in range(m.ROLE)],axis=1).reshape(x.shape)
    mirror=view_decode();native=view_decode()
    assert np.array_equal(mirror,native)
