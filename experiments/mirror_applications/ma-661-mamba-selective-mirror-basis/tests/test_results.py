import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ma661", ROOT / "source" / "run_experiment.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class MA661EvidenceTests(unittest.TestCase):
    def test_grid_actual_bytes_and_native_alias(self):
        rows = json.loads((ROOT / "source" / "development.json").read_text())
        self.assertEqual(len(rows), 96)
        self.assertEqual({r["seed"] for r in rows}, {66101, 66102})
        for row in rows:
            path = ROOT / "source" / f"{row['seed']}_rho{row['residual_rms']:g}_k{row['rank']}_{row['method']}.npz"
            self.assertTrue(path.exists(), path)
            self.assertEqual(path.stat().st_size, row["serialized_bytes"])
            if row["method"] == "mirror_shared_view":
                stem = f"{row['seed']}_rho{row['residual_rms']:g}_k{row['rank']}_"
                a = np.load(ROOT / "source" / f"{stem}mirror_shared_view.npz")
                b = np.load(ROOT / "source" / f"{stem}native_joint_basis.npz")
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])

    def test_development_metric_replay(self):
        rows = json.loads((ROOT / "source" / "development.json").read_text())
        expected = {(r["seed"], r["residual_rms"], r["rank"], r["method"]): r for r in rows}
        for seed in (66101, 66102):
            for rho in (0.0, 0.1, 0.3):
                for rank in (1, 2, 4, 8):
                    for row in RUNNER.run_condition(seed, rho, rank):
                        old = expected[(seed, rho, rank, row["method"])]
                        self.assertEqual(row["serialized_bytes"], old["serialized_bytes"])
                        self.assertAlmostEqual(row["output_nrmse"], old["output_nrmse"], places=12)


if __name__ == "__main__":
    unittest.main()
