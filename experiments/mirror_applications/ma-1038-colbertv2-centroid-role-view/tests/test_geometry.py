import importlib.util
import pathlib

import numpy as np

P = pathlib.Path(__file__).parents[1] / "source" / "run_experiment.py"
SPEC = importlib.util.spec_from_file_location("ma1038", P)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_document_view_equals_inverse_query_view_for_all_token_logits():
    rng = np.random.default_rng(1038)
    q = rng.normal(size=(7, 128)).astype(np.float32)
    d = rng.normal(size=(19, 128)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    for angle in M.ANGLES:
        left = M.role_rotation(d, float(angle)) @ q.T
        right = d @ M.q_inverse(q, float(angle)).T
        np.testing.assert_allclose(left, right, rtol=2e-6, atol=1e-6)


def test_rotation_preserves_norm_and_inverse():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(13, 128)).astype(np.float32)
    y = M.role_rotation(x, 0.173)
    np.testing.assert_allclose(np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(M.role_rotation(y, -0.173), x, rtol=2e-6, atol=2e-6)


def test_quantized_residual_reconstruction_is_finite_and_unit_normalized():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(64, 128)).astype(np.float32)
    c = rng.normal(size=(256, 128)).astype(np.float32)
    scales = np.full(128, 1e-3, dtype=np.float32)
    offsets = np.array([0, len(x)], dtype=np.int64)
    dec, codes, residual = M.encode_index(x, offsets, np.array([len(x)]), c, scales, "unit")
    assert codes.dtype == np.uint8 and residual.dtype == np.int8
    assert np.isfinite(dec).all()
    np.testing.assert_allclose(np.linalg.norm(dec, axis=1), 1.0, rtol=1e-6, atol=1e-6)

