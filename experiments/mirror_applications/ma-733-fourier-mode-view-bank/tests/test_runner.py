import importlib.util
import csv
import io
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).parents[1] / "source" / "run_experiment.py"
spec = importlib.util.spec_from_file_location("ma733", SOURCE)
ma733 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ma733)


def test_problem_shapes_and_alias_reconstruction():
    _, base, atoms, codes, spectra = ma733._make_problem()
    assert base.shape == (64,)
    assert atoms.shape == (4, 64)
    assert codes.shape == (16, 4)
    assert spectra.shape == (3, 16, 64)
    np.testing.assert_allclose(base[None, :] + codes @ atoms, spectra[0], atol=1e-12)


def test_residual_budget_and_serialization_are_paid():
    x = np.arange(64, dtype=float)
    approx, idx, vals = ma733._topk_residual(x, np.zeros_like(x), 4)
    assert len(idx) == len(vals) == 4
    assert np.count_nonzero(approx) == 4
    payload = ma733._npz_bytes(indices=idx, values=vals)
    assert len(payload) > idx.nbytes + vals.nbytes
    with np.load(io.BytesIO(payload)) as restored:
        np.testing.assert_array_equal(restored["indices"], idx)
        np.testing.assert_array_equal(restored["values"], vals)


def test_saved_mirror_rows_match_native_shared_band_exactly():
    result = SOURCE.parents[1] / "RESULTS_CORE.csv"
    rows = list(csv.DictReader(result.open(newline="")))
    fields = ("rho", "private_topk", "residual_selection", "resolution")
    mirror = {tuple(row[f] for f in fields): row for row in rows if row["method"] == "mirror_band_view"}
    native = {tuple(row[f] for f in fields): row for row in rows if row["method"] == "native_shared_band"}
    assert len(mirror) == len(native) == 54
    for key, m in mirror.items():
        n = native[key]
        assert m["payload_sha256"] == n["payload_sha256"]
        assert m["serialized_bytes"] == n["serialized_bytes"]
        assert m["relative_error"] == n["relative_error"]
        assert m["high_band_error"] == n["high_band_error"]
