import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'))
import run
class Checks(unittest.TestCase):
 def test_mask(self):
  _,_,v,m=run.world(25700,0);self.assertEqual(tuple(v.shape),(8,8,256));self.assertEqual(int(m.sum()),48);self.assertEqual(int((~m).sum()),16);self.assertTrue(m.any(0).all());self.assertTrue(m.any(1).all())
 def test_factor_fit(self):
  _,_,v,m=run.world(25700,0);a,b=run.fit_factors(v,m);self.assertLess(run.nrmse(a[:,None,:]+b[None,:,:],v),1e-5)
 def test_metric_shapes(self):
  x=run.torch.randn(32,256);self.assertLess(run.nrmse(x,x),1e-7)
 def test_fresh_locked(self):
  import json;p=json.loads((ROOT/'PROTOCOL.json').read_text());self.assertEqual(p['fresh']['worlds'],[25710,25711,25712])
 def test_fresh_grid_and_payload_hashes(self):
  import csv,hashlib
  rows=list(csv.DictReader((ROOT/'FRESH_RESULTS.csv').open()));self.assertEqual(len(rows),45)
  self.assertEqual({int(r['world']) for r in rows},{25710,25711,25712})
  for r in rows:
   b=(ROOT.parents[2]/r['path']).read_bytes();self.assertEqual(len(b),int(r['payload_bytes']));self.assertEqual(hashlib.sha256(b).hexdigest(),r['sha256'])
 def test_development_artifact_grid_and_hashes(self):
  import csv,hashlib
  rows=list(csv.DictReader((ROOT/'DEVELOPMENT_RESULTS.csv').open()));self.assertEqual(len(rows),30)
  for r in rows:
   b=(ROOT.parents[2]/r['path']).read_bytes();self.assertEqual(len(b),int(r['payload_bytes']));self.assertEqual(hashlib.sha256(b).hexdigest(),r['sha256'])
if __name__=='__main__':unittest.main()
