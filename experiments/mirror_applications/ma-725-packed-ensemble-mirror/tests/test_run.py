import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class PackedViews(unittest.TestCase):
 def test_member_quality_alias_and_replay(self):
  a=run.run(72501);b=run.run(72501)
  self.assertEqual([(x['payload_sha256'],x['member_mse']) for x in a],[(x['payload_sha256'],x['member_mse']) for x in b])
  d={(x['method'],x['member']):x for x in a}
  for m in range(run.M):self.assertAlmostEqual(d['independent',m]['member_mse'],d['packed_independent',m]['member_mse'],places=5)
  self.assertEqual(d['givens',0]['payload_sha256'],d['givens',3]['payload_sha256'])
  self.assertTrue(all(d['givens',m]['member_mse']<.02 for m in range(run.M)))
if __name__=='__main__':unittest.main()
