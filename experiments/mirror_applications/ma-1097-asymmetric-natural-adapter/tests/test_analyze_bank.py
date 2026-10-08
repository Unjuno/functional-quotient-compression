import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "source" / "analyze_bank.py"
SPEC = importlib.util.spec_from_file_location("ma1097_analyze_bank", SCRIPT)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def test_rank_one_gauge_does_not_change_update_or_projectors():
    rng = np.random.default_rng(1097)
    B = rng.normal(size=(9, 1))
    A = rng.normal(size=(1, 7))
    assert ANALYSIS.gauge_diagnostic(B, A) < 1e-12


def test_shared_output_and_input_bases_are_orthonormal_and_deterministic():
    d1 = np.outer(np.arange(1.0, 5.0), np.arange(2.0, 7.0))
    d2 = np.outer(np.arange(4.0, 0.0, -1.0), np.arange(1.0, 6.0))
    u = ANALYSIS.orth_basis([d1, d2], "out", 2)
    v = ANALYSIS.orth_basis([d1, d2], "in", 2)
    assert u.shape == (4, 2)
    assert v.shape == (5, 2)
    assert np.allclose(u.T @ u, np.eye(2), atol=1e-12)
    assert np.allclose(v.T @ v, np.eye(2), atol=1e-12)
    assert np.allclose(u, ANALYSIS.orth_basis([d1, d2], "out", 2))


def test_factorized_basis_matches_dense_shared_subspace():
    rng = np.random.default_rng(97)
    records = []
    mats = []
    for _ in range(2):
        B = rng.normal(size=(8, 1))
        A = rng.normal(size=(1, 6))
        records.append({"A": A, "B": B})
        mats.append(B @ A)
    for side in ("out", "in"):
        dense = ANALYSIS.orth_basis(mats, side, 2)
        factorized = ANALYSIS.orth_basis_from_factors(records, side, 2)
        assert np.allclose(dense @ dense.T, factorized @ factorized.T, atol=1e-12)


def test_relative_error_is_zero_for_exact_reconstruction():
    d = np.arange(12.0).reshape(3, 4)
    assert ANALYSIS.rel_error([d], [d.copy()]) == 0.0
