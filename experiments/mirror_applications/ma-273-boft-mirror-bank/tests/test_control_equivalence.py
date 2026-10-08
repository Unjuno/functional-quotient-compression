import csv
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ControlAudit(unittest.TestCase):
    def test_fresh_mirror_matches_ordinary_factorized_control(self):
        with (ROOT / "RESULTS_CORE.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        fresh = [r for r in rows if r["world_or_seed"] in {"163", "269", "367", "463"}]
        self.assertEqual(len(fresh), 40)
        for condition in {"aligned_angle_orbit", "independent_angle_stress"}:
            for seed in {"163", "269", "367", "463"}:
                pair = {r["method"]: r for r in fresh if r["condition"] == condition and r["world_or_seed"] == seed}
                self.assertEqual(pair["mirror_view"]["serialized_bytes"], pair["factorized_control"]["serialized_bytes"])
                self.assertEqual(pair["mirror_view"]["primary_value"], pair["factorized_control"]["primary_value"])
        aligned = [r for r in fresh if r["condition"] == "aligned_angle_orbit" and r["method"] == "mirror_view"]
        independent = [r for r in fresh if r["condition"] == "aligned_angle_orbit" and r["method"] == "independent_boft"]
        self.assertEqual({r["serialized_bytes"] for r in aligned}, {"1150"})
        self.assertEqual({r["serialized_bytes"] for r in independent}, {"1582"})

if __name__ == "__main__":
    unittest.main()
