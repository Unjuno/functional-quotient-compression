import sys
import unittest
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))
import engine


class MA297Tests(unittest.TestCase):
    def test_payload_roundtrip_and_byte_exact(self):
        records = [("name", "mirror"), ("tensor", np.arange(12, dtype=np.float32).reshape(3, 4))]
        payload = engine.pack(records)
        decoded = engine.unpack(payload)
        self.assertEqual(decoded[0], ("name", b"mirror"))
        self.assertTrue(np.array_equal(decoded[1][1], records[1][1]))
        self.assertEqual(len(payload), len(engine.pack(records)))

    def test_world_generation_is_fresh_and_reproducible(self):
        a = engine.make_world(29711, "aligned", 1)
        b = engine.make_world(29711, "aligned", 1)
        self.assertTrue(np.array_equal(a[2][0], b[2][0]))
        self.assertFalse(np.array_equal(a[2][1], engine.make_world(29711, "independent", 1)[2][1]))

    def test_methods_use_training_data_and_emit_paid_payloads(self):
        rows = engine.run(29711, "fresh", 0.0, 1)
        self.assertEqual(len(rows), 2 * 5 * 5)
        mirror = [r for r in rows if r["condition"] == "aligned" and r["method"] == "mirror"]
        coeff = [r for r in rows if r["condition"] == "aligned" and r["method"] == "coeff2"]
        self.assertTrue(all(r["inference_payload_bytes"] > 0 for r in mirror + coeff))
        self.assertEqual(mirror[-1]["optimizer_updates_cumulative"], 0)
        self.assertGreater(mirror[-1]["active_compute_proxy"], coeff[-1]["active_compute_proxy"])
        self.assertLess(max(r["reconstruction_max_abs_diff"] for r in rows), 3e-7)
        independent=[r for r in rows if r["method"]=="independent_full" and r["tasks_seen"]==5]
        self.assertLess(independent[0]["inference_payload_bytes"], 2200)

    def test_seed_replay(self):
        rows = engine.deterministic_rows(29711, "fresh", 0.0, 1)
        self.assertEqual(len(rows), 50)


if __name__ == "__main__":
    unittest.main()
