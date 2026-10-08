import hashlib,importlib.util,json,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[4];EXP=ROOT/'experiments/mirror_applications/ma-439-packet-plan-ssm-mirror'
spec=importlib.util.spec_from_file_location('ma439',EXP/'source'/'run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
class ArtifactTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.rows=[json.loads(x) for x in (EXP/'artifacts'/'fresh_runs.jsonl').read_text().splitlines()]
 def test_fresh_payloads_roundtrip_hash(self):
  self.assertEqual(len(self.rows),36);seen=set()
  for r in self.rows:
   key=(r['method'],r['world_or_seed'])
   if key in seen:continue
   seen.add(key);f=dict(x.split('=',1) for x in r['status_note'].split('; ') if '=' in x);b=(ROOT/f['payload']).read_bytes()
   self.assertEqual(len(b),r['serialized_bytes']);self.assertEqual(hashlib.sha256(b).hexdigest(),f['hash']);obj=torch.load(ROOT/f['payload'],map_location='cpu',weights_only=False);self.assertEqual(obj['method'],r['method'])
 def test_metric_replay_stability(self):
  for r in self.rows:
   f=dict(x.split('=',1) for x in r['status_note'].split('; ') if '=' in x);m,d=run.replay(ROOT/f['payload']);w=int(r['world_or_seed'].split('-')[0]);v,_,rho=run.evaluate(m,w);self.assertAlmostEqual(v,r['primary_value'],places=8);self.assertAlmostEqual(rho,r['secondary_value'],places=8);self.assertLess(rho,1)
if __name__=='__main__':unittest.main()
