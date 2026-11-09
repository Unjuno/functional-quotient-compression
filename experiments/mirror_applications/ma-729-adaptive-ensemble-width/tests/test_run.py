import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class AdaptiveWidth(unittest.TestCase):
 def test_replay_and_full_width(self):
  a=run.run(72901);b=run.run(72901)
  self.assertEqual([(x['payload_sha256'],x['accuracy']) for x in a],[(x['payload_sha256'],x['accuracy']) for x in b])
  d={x['method']:x for x in a}
  self.assertAlmostEqual(d['fixed4']['accuracy'],d['native_angle_bank']['accuracy'],places=12)
  self.assertAlmostEqual(d['fixed4']['nll'],d['native_angle_bank']['nll'],places=12)
  self.assertEqual(len(a),13)
if __name__=='__main__':unittest.main()
