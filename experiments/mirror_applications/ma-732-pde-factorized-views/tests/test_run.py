import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class PDEViews(unittest.TestCase):
 def test_native_alias_accuracy_and_replay(self):
  a=run.run(73201);b=run.run(73201)
  self.assertEqual([(x['payload_sha256'],x['heldout_relative_l2']) for x in a],[(x['payload_sha256'],x['heldout_relative_l2']) for x in b])
  d={(x['method'],x['resolution']):x for x in a}
  for n in (32,64):
   self.assertEqual(d['native_parameter_conditioning',n]['payload_sha256'],d['Mirror_factor_code',n]['payload_sha256'])
   self.assertLess(d['Mirror_factor_code',n]['heldout_relative_l2'],1e-10)
  self.assertEqual(len(a),8)
if __name__=='__main__':unittest.main()
