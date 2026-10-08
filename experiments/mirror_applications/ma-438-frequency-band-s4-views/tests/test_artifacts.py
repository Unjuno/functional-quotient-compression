import hashlib,importlib.util,json,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[4];EXP=ROOT/'experiments/mirror_applications/ma-438-frequency-band-s4-views'
spec=importlib.util.spec_from_file_location('ma438',EXP/'source'/'run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
class ArtifactTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.rows=[json.loads(s) for s in (EXP/'artifacts'/'fresh_runs.jsonl').read_text().splitlines()]
 def test_payload_hash_bytes_and_roundtrip(self):
  self.assertEqual(len(self.rows),90);seen=set()
  for r in self.rows:
   key=(r['method'],r['world_or_seed'].rsplit('-',1)[0])
   if key in seen:continue
   seen.add(key);f=dict(x.split('=',1) for x in r['status_note'].split('; ') if '=' in x);b=(ROOT/f['payload']).read_bytes()
   self.assertEqual(len(b),r['serialized_bytes']);self.assertEqual(hashlib.sha256(b).hexdigest(),f['hash'])
   obj=torch.load(ROOT/f['payload'],map_location='cpu',weights_only=False);self.assertEqual(obj['method'],r['method'])
 def test_metrics_replay_and_stability(self):
  for r in self.rows:
   f=dict(x.split('=',1) for x in r['status_note'].split('; ') if '=' in x);m,d=run.replay(ROOT/f['payload']);w,s,L=map(int,r['world_or_seed'].split('-'))
   score,_,rho=run.metric(m,w,L);self.assertAlmostEqual(score,r['primary_value'],places=8);self.assertAlmostEqual(rho,r['secondary_value'],places=8);self.assertLess(rho,1)
if __name__=='__main__':unittest.main()
