import hashlib,importlib.util,json,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[4];EXP=ROOT/'experiments/mirror_applications/ma-441-state-space-address-memory'
spec=importlib.util.spec_from_file_location('ma441',EXP/'source'/'run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
class ArtifactTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.rows=[json.loads(x) for x in (EXP/'artifacts'/'fresh_runs.jsonl').read_text().splitlines()]
 def test_payload_bytes_hashes_and_state_bytes(self):
  self.assertEqual(len(self.rows),144);seen=set()
  for r in self.rows:
   key=(r['method'],r['world_or_seed'].rsplit('-',1)[0])
   if key in seen:continue
   seen.add(key);f=dict(x.split('=',1) for x in r['status_note'].split('; ') if '=' in x);b=(ROOT/f['payload']).read_bytes();self.assertEqual(len(b),r['serialized_bytes']);self.assertEqual(hashlib.sha256(b).hexdigest(),f['hash']);self.assertEqual(int(f['state_bytes']),run.dimension(r['method'])*4)
 def test_metric_replay(self):
  for r in self.rows:
   f=dict(x.split('=',1) for x in r['status_note'].split('; ') if '=' in x);m=run.Memory(r['method']);d=torch.load(ROOT/f['payload'],map_location='cpu',weights_only=False);m.load_state_dict(d['state']);w,s,L=map(int,r['world_or_seed'].split('-'));score,acc,_,drift=run.evaluate(m,w,L,w*100+s+L);self.assertAlmostEqual(score,r['primary_value'],places=8);self.assertAlmostEqual(acc,r['secondary_value'],places=8);self.assertAlmostEqual(drift,float(f['drift']),places=8)
if __name__=='__main__':unittest.main()
