import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class PosteriorViews(unittest.TestCase):
 def test_alias_metrics_and_replay(self):
  a=run.run(72101);b=run.run(72101);self.assertEqual([(x['payload_sha256'],x['predictive_nll']) for x in a],[(x['payload_sha256'],x['predictive_nll']) for x in b])
  d={}
  for x in a:d.setdefault((x['rank'],x['method']),x)
  for k in (1,2,4,8,16):self.assertEqual(d[k,'SWAG_lowrank_diag']['payload_sha256'],d[k,'Mirror_posterior_coordinate']['payload_sha256'])
  self.assertEqual(len(a),12)
if __name__=='__main__':unittest.main()
