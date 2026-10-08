import csv
import unittest
from pathlib import Path


class PreregisteredGateAudit(unittest.TestCase):
    def test_relative_mse_gate_is_not_met_in_three_of_four_fresh_worlds(self):
        path = Path(__file__).resolve().parents[1] / "RESULTS_CORE.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        ratios = []
        for seed in ("107", "227", "311", "419"):
            independent = next(
                float(row["primary_value"])
                for row in rows
                if row["world_or_seed"] == seed
                and row["condition"] == "aligned_givens"
                and row["method"] == "independent"
            )
            mirror = next(
                float(row["primary_value"])
                for row in rows
                if row["world_or_seed"] == seed
                and row["condition"] == "aligned_givens"
                and row["method"] == "mirror_givens"
            )
            ratios.append(mirror / independent)

        passing_worlds = sum(ratio <= 1.10 for ratio in ratios)
        self.assertEqual(passing_worlds, 1)
        self.assertLess(passing_worlds, 3)  # preregistered gate requires at least 3/4


if __name__ == "__main__":
    unittest.main()
