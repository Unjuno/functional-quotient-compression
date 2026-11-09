import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class PrivateFrontier(unittest.TestCase):
 def test_frontier_and_replay(self):
  a=run.run(71901);b=run.run(71901)
  self.assertEqual([(x['payload_sha256'],x['heldout_nrmse']) for x in a],[(x['payload_sha256'],x['heldout_nrmse']) for x in b])
  self.assertEqual(len(a),4*(2+2*len(run.KS)))
  self.assertTrue(all(x['serialized_bytes']>0 for x in a))
if __name__=='__main__':unittest.main()
