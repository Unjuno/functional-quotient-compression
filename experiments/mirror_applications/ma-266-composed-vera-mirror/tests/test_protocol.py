import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'));import run
class Checks(unittest.TestCase):
 def test_idempotent_composition_law(self):
  u,a,I,K,W1,W2,Wc,ac=run.bank(26600,0);self.assertLess(float((K@K-K).abs().max()),1e-6);self.assertLess(run.nrmse(W1@W2,I+ac*K),1e-6)
 def test_shapes(self):
  self.assertEqual(tuple(run.inputs(26600,0).shape),(1024,32))
 def test_binary_roundtrip(self):
  u,a,I,K,W1,W2,Wc,ac=run.bank(26600,0);blob=run.pack('mirror_composed',u,ac.reshape(1),Wc);name,state=run.decode(blob);self.assertEqual(name,'mirror_composed');self.assertLess(run.nrmse(state['W'],Wc),1e-6);self.assertEqual(len(blob),140)
 def test_dev_payload_hashes_and_sizes(self):
  import csv,hashlib
  rows=list(csv.DictReader((ROOT/'DEVELOPMENT_RESULTS.csv').open()));self.assertEqual(len(rows),36)
  for r in rows:
   b=(ROOT.parents[2]/r['path']).read_bytes();self.assertEqual(len(b),int(r['payload_bytes']));self.assertEqual(hashlib.sha256(b).hexdigest(),r['sha256'])
 def test_fresh_lock(self):
  p=json.loads((ROOT/'PROTOCOL.json').read_text());self.assertTrue(p['fresh']['locked_before_access']);self.assertEqual(p['fresh']['worlds'],[26610,26611,26612])
if __name__=='__main__':unittest.main()
