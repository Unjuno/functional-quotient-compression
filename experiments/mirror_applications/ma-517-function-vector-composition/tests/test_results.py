import csv
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]


class ResultChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "RESULTS_CORE.csv").open() as f:
            cls.rows = list(csv.DictReader(f))

    def test_full_world_seed_method_grid(self):
        self.assertEqual(len(self.rows), 54)
        self.assertEqual({int(r["world"]) for r in self.rows}, {51710, 51711, 51712})
        self.assertEqual({int(r["seed"]) for r in self.rows}, {0, 1, 2})
        self.assertEqual({r["method"] for r in self.rows}, {"query_only", "direct_icl", "raw_sum", "factorized", "f1_only", "f2_only"})

    def test_all_accuracy_zero_and_baseline_gate_failure(self):
        self.assertTrue(all(float(r["accuracy"]) == 0.0 for r in self.rows))
        self.assertTrue(all(float(r["accuracy"]) == 0.0 for r in self.rows if r["method"] == "direct_icl"))

    def test_payload_serialization_bytes_and_hashes(self):
        payload_rows = [r for r in self.rows if r["method"] in {"raw_sum", "factorized", "f1_only", "f2_only"}]
        self.assertEqual(len(payload_rows), 36)
        for row in payload_rows:
            blob = (REPO / row["path"]).read_bytes()
            self.assertEqual(len(blob), int(row["payload_bytes"]))
            self.assertEqual(hashlib.sha256(blob).hexdigest(), row["hash"])

    def test_fresh_split_matches_frozen_protocol(self):
        protocol = json.loads((ROOT / "PROTOCOL.json").read_text())
        self.assertEqual(protocol["fresh"]["worlds"], [51710, 51711, 51712])
        self.assertEqual(protocol["fresh"]["seeds"], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
