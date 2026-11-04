import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class ResultIntegrityTests(unittest.TestCase):
    def test_complete_seed_grid_and_actual_serialized_sizes(self):
        for filename, expected_seeds in (("development.json", {65501, 65502}), ("fresh.json", {65511, 65512, 65513})):
            rows = json.loads((ROOT / "source" / filename).read_text())
            self.assertEqual(len(rows), len(expected_seeds) * 2 * 3 * 5)
            self.assertEqual({r["seed"] for r in rows}, expected_seeds)
            for row in rows:
                payload = ROOT / "source" / f"{row['seed']}_n{row['program_count']}_rho{row['private_function_residual_rms']:g}_{row['method']}.npz"
                self.assertTrue(payload.exists(), payload)
                self.assertEqual(payload.stat().st_size, row["serialized_bytes"])

    def test_mirror_and_native_pca_are_identical(self):
        rows = json.loads((ROOT / "source" / "fresh.json").read_text())
        for row in rows:
            if row["method"] != "mirror_pca_rank4":
                continue
            stem = f"{row['seed']}_n{row['program_count']}_rho{row['private_function_residual_rms']:g}_"
            mirror = np.load(ROOT / "source" / f"{stem}mirror_pca_rank4.npz")
            native = np.load(ROOT / "source" / f"{stem}native_pca_rank4.npz")
            self.assertEqual(set(mirror.files), set(native.files))
            for key in mirror.files:
                np.testing.assert_array_equal(mirror[key], native[key])


if __name__ == "__main__":
    unittest.main()
