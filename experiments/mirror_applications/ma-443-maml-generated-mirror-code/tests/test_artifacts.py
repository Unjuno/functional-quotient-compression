import ast,hashlib,importlib.util,json,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[4];EXP=ROOT/'experiments/mirror_applications/ma-443-maml-generated-mirror-code'
spec=importlib.util.spec_from_file_location('ma443',EXP/'source'/'run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
class ArtifactTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.rows=[json.loads(s) for s in (EXP/'artifacts'/'fresh_runs.jsonl').read_text().splitlines()]
 def test_fresh_task_payloads_exact(self):
  self.assertEqual(len(self.rows),3600)
  for r in self.rows:
   d=dict(x.split('=',1) for x in r['status_note'].split('; ') if '=' in x);p=ROOT/d['payload'];b=p.read_bytes();self.assertEqual(len(b),r['serialized_bytes']);self.assertEqual(hashlib.sha256(b).hexdigest(),d['hash'])
   obj=torch.load(p,map_location='cpu',weights_only=False);self.assertEqual(obj['method'],r['method'])
 def test_encoder_amortization_and_metric_replay(self):
  enc_seen=set()
  for r in self.rows:
   d=dict(x.split('=',1) for x in r['status_note'].split('; ') if '=' in x);obj=torch.load(ROOT/d['payload'],map_location='cpu',weights_only=False);w,seed,task,steps=map(int,r['world_or_seed'].split('-'));x,y=run.samples(w,task,128,seed+100+771)
   with torch.no_grad():err=((x@run.eff(r['method'],obj['base'],obj['code'],obj['basis'])-y).square().mean().sqrt()/(y.square().mean().sqrt()+1e-12)).item()
   self.assertAlmostEqual(err,r['primary_value'],places=8)
   am=ast.literal_eval(d['amortized_bytes']);self.assertEqual(am[20],round(r['serialized_bytes']+int(d['encoder_bytes'])/20))
   if 'encoder_path' in d:
    ep=ROOT/d['encoder_path'];eb=ep.read_bytes();self.assertEqual(len(eb),int(d['encoder_bytes']));self.assertEqual(hashlib.sha256(eb).hexdigest(),d['encoder_hash']);enc_seen.add(str(ep))
  self.assertGreater(len(enc_seen),0)
if __name__=='__main__':unittest.main()
