import tempfile,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class LDRTests(unittest.TestCase):
 def test_replay_bytes_and_native_alias(self):
  with tempfile.TemporaryDirectory() as d:
   rows=run.run_one('audit-70811',70811,Path(d))
   for r in rows:
    self.assertLessEqual(r['heldout_operator_nrmse'],1e-5)
    self.assertEqual(r['payload_sha256'],r['native_sha256'])
    self.assertEqual(r['serialized_bytes'],r['native_bytes'])
    if r['rank'] < 8: self.assertLess(r['serialized_bytes'],r['independent_bytes'])
    else: self.assertGreater(r['serialized_bytes'],r['independent_bytes'])
if __name__=='__main__':unittest.main()
