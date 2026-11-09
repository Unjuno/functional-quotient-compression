import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class CapabilityAtoms(unittest.TestCase):
 def test_alias_and_replay(self):
  a=run.run(71801);b=run.run(71801)
  self.assertEqual([(r['payload_sha256'],r['heldout_nrmse']) for r in a],[(r['payload_sha256'],r['heldout_nrmse']) for r in b])
  x={(r['method'],r['rank']):r for r in a}
  for k in (2,4,6,8,12,16):self.assertEqual(x['native_PCA',k]['payload_sha256'],x['Mirror_atoms',k]['payload_sha256'])
  self.assertEqual(len(a),15)
if __name__=='__main__':unittest.main()
