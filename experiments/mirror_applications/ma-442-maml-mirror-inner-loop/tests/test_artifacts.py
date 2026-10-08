import hashlib,importlib.util,json,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[4];EXP=ROOT/'experiments/mirror_applications/ma-442-maml-mirror-inner-loop'
spec=importlib.util.spec_from_file_location('ma442',EXP/'source'/'run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
class ArtifactTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.rows=[json.loads(x) for x in (EXP/'artifacts'/'fresh_runs.jsonl').read_text().splitlines()]
 def test_serialized_payloads_and_hashes(self):
  self.assertEqual(len(self.rows),720)
  for r in self.rows:
   f=dict(x.split('=',1) for x in r['status_note'].split('; ') if '=' in x);p=ROOT/f['payload'];b=p.read_bytes();self.assertEqual(len(b),r['serialized_bytes']);self.assertEqual(hashlib.sha256(b).hexdigest(),f['hash']);obj=torch.load(p,map_location='cpu',weights_only=False);self.assertEqual(obj['method'],r['method'])
 def test_metric_replay(self):
  for r in self.rows:
   f=dict(x.split('=',1) for x in r['status_note'].split('; ') if '=' in x);d=torch.load(ROOT/f['payload'],map_location='cpu',weights_only=False);w,s,t=map(int,r['world_or_seed'].split('-'));err,_,_,_,_=run.evaluate(d['base'],d['basis'],d['method'],w,t,100+s,run.ILR);self.assertAlmostEqual(err,r['primary_value'],places=8)
if __name__=='__main__':unittest.main()
