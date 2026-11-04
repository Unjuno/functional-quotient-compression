import json
from pathlib import Path

import numpy as np

from run_experiment import ROOT, fit_alpha, make_world


def test_weighted_closed_form_is_coordinatewise_optimal():
    rng = np.random.default_rng(64999)
    g = rng.normal(size=(200, 7)); y = rng.normal(size=(200, 7)); lam = np.exp(rng.normal(size=7))
    alpha = fit_alpha(g, y, lam)
    expected = np.sum(lam * g * y, axis=0) / np.sum(lam * g * g, axis=0)
    assert np.allclose(alpha, expected)


def test_development_grid_actual_bytes_and_mirror_native_alias():
    rows = json.loads((ROOT / "source" / "development.json").read_text())
    assert len(rows) == 30
    for row in rows:
        path = ROOT / "source" / f"{row['seed']}_h{row['heterogeneity']:g}_{row['method']}.npz"
        assert path.stat().st_size == row["serialized_bytes"]
    for seed in (64901, 64902):
        for h in (0., .25, .5):
            p1 = ROOT / "source" / f"{seed}_h{h:g}_mirror_pca_rank4.npz"
            p2 = ROOT / "source" / f"{seed}_h{h:g}_native_pca_rank4.npz"
            with np.load(p1) as a, np.load(p2) as b:
                for key in ("base_initialization", "family_ids", "shared_basis", "family_codes"):
                    assert np.array_equal(a[key], b[key])


def test_query_metric_replay():
    rows = json.loads((ROOT / "source" / "development.json").read_text())
    for row in rows:
        seed, h, method = row["seed"], row["heterogeneity"], row["method"]
        _base, lambdas, _train, query = make_world(seed, h)
        p = ROOT / "source" / f"{seed}_h{h:g}_{method}.npz"
        with np.load(p) as payload:
            if method == "global_meta_sgd":
                alpha = np.repeat(payload["global_update_vector"][None, :], 8, axis=0)
            elif method == "independent_family_meta_sgd":
                alpha = payload["family_update_vectors"]
            elif method == "mirror_pca_rank4" or method == "native_pca_rank4":
                alpha = payload["family_codes"] @ payload["shared_basis"].T
            else:
                alpha = payload["family_codes"] @ payload["random_basis"].T
        losses = []
        for f, (g, y) in enumerate(query):
            losses.append(float(.5 * np.mean(lambdas[f][None, :] * (alpha[f][None, :] * g - y) ** 2)))
        assert abs(float(np.mean(losses)) - row["mean_query_loss"]) < 1e-7
        assert abs(float(np.max(losses)) - row["max_family_query_loss"]) < 1e-7
