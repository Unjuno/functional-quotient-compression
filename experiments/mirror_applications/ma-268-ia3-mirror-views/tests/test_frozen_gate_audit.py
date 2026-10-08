import csv
import unittest
from pathlib import Path


class FrozenGateAudit(unittest.TestCase):
    def test_nonlinear_fresh_results_satisfy_registered_gate(self):
        path = Path(__file__).resolve().parents[1] / "RESULTS_CORE.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        for seed in ("26802", "26803", "26804"):
            fresh = [row for row in rows if row["world_or_seed"] == seed]
            by_method = {row["method"]: row for row in fresh}
            mirror = by_method["mirror"]
            ia3 = by_method["ia3"]
            independent = by_method["independent"]

            self.assertLessEqual(
                float(mirror["mean_task_mse"]),
                1.10 * float(independent["mean_task_mse"]),
                seed,
            )
            self.assertLessEqual(
                int(mirror["serialized_bytes"]),
                0.65 * int(independent["serialized_bytes"]),
                seed,
            )
            self.assertLess(float(mirror["mean_task_mse"]), float(ia3["mean_task_mse"]), seed)
            self.assertLess(int(mirror["serialized_bytes"]), int(ia3["serialized_bytes"]), seed)


if __name__ == "__main__":
    unittest.main()
