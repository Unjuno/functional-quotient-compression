import csv
import statistics
import unittest
from pathlib import Path


class ScalarControlAudit(unittest.TestCase):
    def test_scalar_generator_matches_mirror_and_latency_gate_fails(self):
        path = Path(__file__).resolve().parents[1] / "RESULTS_CORE.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        mirror_times = []
        materialized_times = []
        for seed in ("157", "263", "359", "461"):
            for condition in ("shared_generator_orbit", "independent_skew_stress"):
                by_method = {
                    row["method"]: row
                    for row in rows
                    if row["world_or_seed"] == seed and row["condition"] == condition
                }
                mirror = by_method["shared_mirror"]
                simple = by_method["simple_scalar_generator"]
                self.assertEqual(mirror["serialized_bytes"], simple["serialized_bytes"])
                self.assertEqual(mirror["primary_value"], simple["primary_value"])
                self.assertEqual(mirror["secondary_value"], simple["secondary_value"])

            aligned = {
                row["method"]: row
                for row in rows
                if row["world_or_seed"] == seed and row["condition"] == "shared_generator_orbit"
            }
            mirror_times.append(float(aligned["shared_mirror"]["wall_time_s"]))
            materialized_times.append(float(aligned["materialized_oft"]["wall_time_s"]))

        ratio = statistics.mean(mirror_times) / statistics.mean(materialized_times)
        self.assertGreater(ratio, 1.0)  # preregistered runtime gate required <=0.90


if __name__ == "__main__":
    unittest.main()
