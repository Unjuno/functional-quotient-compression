import hashlib,importlib.util,json,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[4];EXP=ROOT/'experiments/mirror_applications/ma-440-continual-ssm-skill-codes'
spec=importlib.util.spec_from_file_location('ma440',EXP/'source'/'run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
class ArtifactTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.rows=[json.loads(x) for x in (EXP/'artifacts'/'fresh_runs.jsonl').read_text().splitlines()]
 def test_fresh_payload_bytes_and_hashes(self):
  self.assertEqual(len(self.rows),144);seen=set()
  for r in self.rows:
   key=(r['method'],r['world_or_seed'].rsplit('-skill',1)[0])
   if key in seen:continue
   seen.add(key);f=dict(x.split('=',1) for x in r['status_note'].split('; ') if '=' in x);b=(ROOT/f['payload']).read_bytes();self.assertEqual(len(b),r['serialized_bytes']);self.assertEqual(hashlib.sha256(b).hexdigest(),f['hash'])
 def test_metrics_replay_and_stability(self):
  for r in self.rows:
   f=dict(x.split('=',1) for x in r['status_note'].split('; ') if '=' in x);m,d=run.replay(ROOT/f['payload']);parts=r['world_or_seed'].split('-');w=int(parts[0]);k=int(parts[2].replace('skill',''));err,_,rho=run.score(m,w,k);self.assertAlmostEqual(err,r['primary_value'],places=8);self.assertAlmostEqual(rho,r['secondary_value'],places=8);self.assertLess(rho,1)
if __name__=='__main__':unittest.main()
