import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ma658", ROOT / "source" / "run_experiment.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class MA658EvidenceTests(unittest.TestCase):
    def test_all_rows_match_actual_payload_sizes(self):
        rows = json.loads((ROOT / "source" / "development.json").read_text())
        self.assertEqual(len(rows), 120)
        self.assertEqual({row["seed"] for row in rows}, {65801, 65802})
        for row in rows:
            path = ROOT / "source" / f"{row['seed']}_n{row['task_count']}_rho{row['residual_rms']:g}_t{row['tasks_written']}_{row['method']}.npz"
            self.assertTrue(path.exists(), path)
            self.assertEqual(path.stat().st_size, row["serialized_bytes"])

    def test_fresh_sealed_native_alias_and_metric_replay(self):
        rows = json.loads((ROOT / "source" / "development.json").read_text())
        expected = {(r["seed"], r["task_count"], r["residual_rms"], r["tasks_written"], r["method"]): r for r in rows}
        for seed in (65801, 65802):
            for n in (8, 32):
                replay = RUNNER.run_condition(seed, n, 0.0)
                for row in replay:
                    key = (seed, n, 0.0, row["tasks_written"], row["method"])
                    old = expected[key]
                    self.assertEqual(row["serialized_bytes"], old["serialized_bytes"])
                    self.assertAlmostEqual(row["mean_task_nrmse"], old["mean_task_nrmse"], places=12)
                    self.assertAlmostEqual(row["worst_task_nrmse"], old["worst_task_nrmse"], places=12)
        for row in rows:
            if row["method"] != "mirror":
                continue
            stem = f"{row['seed']}_n{row['task_count']}_rho{row['residual_rms']:g}_t{row['tasks_written']}_"
            mirror = np.load(ROOT / "source" / f"{stem}mirror.npz")
            native = np.load(ROOT / "source" / f"{stem}native.npz")
            for key in mirror.files:
                np.testing.assert_array_equal(mirror[key], native[key])


if __name__ == "__main__":
    unittest.main()
