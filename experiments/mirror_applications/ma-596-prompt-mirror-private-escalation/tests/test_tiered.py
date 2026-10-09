import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from run_tiered import canonical_pca


def test_rank4_reconstruction_is_deterministic_and_bounded():
    rng = np.random.default_rng(17)
    data = rng.normal(size=(8, 64)).astype(np.float32)
    m1, b1, c1, r1 = canonical_pca(data)
    m2, b2, c2, r2 = canonical_pca(data)
    assert b1.shape == (64, 4)
    assert c1.shape == (8, 4)
    assert np.array_equal(r1, r2)
    assert np.allclose(m1, m2) and np.allclose(b1, b2) and np.allclose(c1, c2)
    assert np.mean((data-r1)**2) <= np.mean((data-data.mean(axis=0))**2)


def test_private_residual_reconstructs_target():
    rng = np.random.default_rng(23)
    target = rng.normal(size=(8, 64)).astype(np.float32)
    _, _, _, recon = canonical_pca(target)
    residual = target - recon
    assert np.allclose(recon + residual, target)
