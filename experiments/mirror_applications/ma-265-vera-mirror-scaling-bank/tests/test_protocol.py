import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'));import run
class Checks(unittest.TestCase):
 def test_shapes(self):
  p,w=run.tasks(26500,0,'aligned');self.assertEqual(tuple(w.shape),(8,32,32));self.assertEqual(tuple(p[10].shape),(8,8))
 def test_code_spaces_finite(self):
  p,w=run.tasks(26500,0,'independent');self.assertTrue(bool(run.torch.isfinite(w).all()))
 def test_dev_grid_payload_hashes(self):
  import csv,hashlib
  rows=list(csv.DictReader((ROOT/'DEVELOPMENT_RESULTS.csv').open()));self.assertEqual(len(rows),48)
  for r in rows:
   b=(ROOT.parents[2]/r['path']).read_bytes();self.assertEqual(len(b),int(r['payload_bytes']));self.assertEqual(hashlib.sha256(b).hexdigest(),r['sha256'])
 def test_fresh_locked(self):
  p=json.loads((ROOT/'PROTOCOL.json').read_text());self.assertTrue(p['fresh']['locked_before_access']);self.assertEqual(p['fresh']['worlds'],[26510,26511,26512])
if __name__=='__main__':unittest.main()
