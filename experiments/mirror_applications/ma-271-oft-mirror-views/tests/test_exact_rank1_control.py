import csv
import unittest
from pathlib import Path


class ExactRankOneControl(unittest.TestCase):
    def test_mirror_and_simple_rank1_have_identical_bytes_and_metrics(self):
        path = Path(__file__).resolve().parents[1] / "RESULTS_CORE.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        for seed in ("127", "239", "331", "443"):
            for condition in ("aligned_orbit", "independent_plane_angles"):
                by_method = {
                    row["method"]: row
                    for row in rows
                    if row["world_or_seed"] == seed and row["condition"] == condition
                }
                mirror = by_method["mirror_view"]
                simple = by_method["simple_rank1_angle"]
                self.assertEqual(mirror["serialized_bytes"], simple["serialized_bytes"])
                self.assertEqual(mirror["primary_value"], simple["primary_value"])
                self.assertEqual(mirror["secondary_value"], simple["secondary_value"])


if __name__ == "__main__":
    unittest.main()
