import hashlib, importlib.util, json, unittest
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[4]
EXP=ROOT/'experiments/mirror_applications/ma-437-s4-structured-transition-mirror'
spec=importlib.util.spec_from_file_location('ma437run',EXP/'source'/'run.py')
run=importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
class ArtifactTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.rows=[json.loads(s) for s in (EXP/'artifacts'/'fresh_runs.jsonl').read_text().splitlines()]
 def test_fresh_payload_hash_bytes_and_roundtrip(self):
  checked=set()
  for row in self.rows:
   if row['method']+'-'+row['world_or_seed'].rsplit('-',1)[0] in checked: continue
   checked.add(row['method']+'-'+row['world_or_seed'].rsplit('-',1)[0])
   fields=dict(x.split('=',1) for x in row['status_note'].split('; ') if '=' in x)
   data=(ROOT/fields['payload']).read_bytes()
   self.assertEqual(len(data),row['serialized_bytes'])
   self.assertEqual(hashlib.sha256(data).hexdigest(),fields['sha256'])
   loaded=torch.load(ROOT/fields['payload'],map_location='cpu',weights_only=False)
   self.assertEqual(loaded['method'],row['method'])
   self.assertEqual(loaded['world'],int(row['world_or_seed'].split('-')[0]))
 def test_fresh_rows_stability_and_replay(self):
  self.assertEqual(len(self.rows),90)
  for row in self.rows:
   self.assertLess(row['secondary_value'],1.0)
   fields=dict(x.split('=',1) for x in row['status_note'].split('; ') if '=' in x)
   model=run.replay_payload(ROOT/fields['payload'])
   world,seed,length=map(int,row['world_or_seed'].split('-'))
   metric,_,rho=run.evaluate(model,world,length)
   self.assertAlmostEqual(metric,row['primary_value'],places=8)
   self.assertAlmostEqual(rho,row['secondary_value'],places=8)
if __name__=='__main__': unittest.main()
