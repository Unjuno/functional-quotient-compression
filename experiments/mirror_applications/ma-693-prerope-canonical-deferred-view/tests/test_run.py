import tempfile
import unittest
from pathlib import Path
import sys, hashlib
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import run

class DeferredRoPE(unittest.TestCase):
    def test_audit_seed_replay_and_boundaries(self):
        with tempfile.TemporaryDirectory() as d:
            r=run.run_one("audit-69311",69311,Path(d))
            self.assertLessEqual(r["max_lazy_rel_error"],1e-5)
            self.assertGreater(r["max_bad_query_shortcut_rel_error"],1e-2)
            self.assertEqual(r["max_native_givens_rel_error"],r["max_lazy_rel_error"])
            p=Path(d)/"audit-69311.lazy.npz"
            self.assertEqual(r["serialized_bytes"],p.stat().st_size)
            self.assertEqual(r["lazy_sha256"],hashlib.sha256(p.read_bytes()).hexdigest())

if __name__=="__main__": unittest.main()
