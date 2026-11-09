import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
import run

class AlgebraAndSerialization(unittest.TestCase):
    def test_locked_seed_replay_and_measurements(self):
        with tempfile.TemporaryDirectory() as d:
            row = run.run_one('audit-69211', 69211, Path(d))
            self.assertLessEqual(row['max_lazy_rel_error'], 1e-5)
            self.assertLessEqual(row['max_native_phase_rel_error'], 1e-5)
            self.assertGreater(row['max_noncommuting_rel_error'], 0.01)
            self.assertEqual(row['serialized_bytes'], (Path(d)/'audit-69211.lazy.npz').stat().st_size)
            self.assertEqual(row['independent_bytes'], (Path(d)/'audit-69211.independent.npz').stat().st_size)
            self.assertEqual(row['lazy_sha256'], __import__('hashlib').sha256((Path(d)/'audit-69211.lazy.npz').read_bytes()).hexdigest())

if __name__ == '__main__':
    unittest.main()
