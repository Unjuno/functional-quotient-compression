import ast,hashlib,importlib.util,json,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[4];EXP=ROOT/'experiments/mirror_applications/ma-444-leo-mirror-decoder'
spec=importlib.util.spec_from_file_location('ma444',EXP/'source'/'run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
class ArtifactTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.rows=[json.loads(s) for s in (EXP/'artifacts'/'fresh_runs.jsonl').read_text().splitlines()]
 def test_task_and_decoder_payload_hashes(self):
  self.assertEqual(len(self.rows),2880);decoders=set()
  for r in self.rows:
   d=dict(x.split('=',1) for x in r['status_note'].split('; ') if '=' in x);p=ROOT/d['payload'];b=p.read_bytes();self.assertEqual(len(b),r['serialized_bytes']);self.assertEqual(hashlib.sha256(b).hexdigest(),d['hash'])
   obj=torch.load(p,map_location='cpu',weights_only=False);self.assertEqual(obj['method'],r['method'])
   am=ast.literal_eval(d['amortized_bytes']);self.assertEqual(am[20],round(r['serialized_bytes']+int(d['decoder_bytes'])/20))
   if 'decoder_path' in d:
    q=ROOT/d['decoder_path'];qb=q.read_bytes();self.assertEqual(len(qb),int(d['decoder_bytes']));self.assertEqual(hashlib.sha256(qb).hexdigest(),d['decoder_hash']);decoders.add(str(q))
  self.assertEqual(len(decoders),9)
 def test_metric_replay(self):
  for r in self.rows:
   d=dict(x.split('=',1) for x in r['status_note'].split('; ') if '=' in x);obj=torch.load(ROOT/d['payload'],map_location='cpu',weights_only=False);basis=torch.load(ROOT/d['decoder_path'],map_location='cpu',weights_only=False)['state'] if 'decoder_path' in d else torch.zeros(run.D,2);w,s,task,steps=map(int,r['world_or_seed'].split('-'));x,y=run.samples(w,task,128,s+100+771)
   with torch.no_grad():e=((x@run.effective(r['method'],obj['base'],obj['code'],basis)-y).square().mean().sqrt()/(y.square().mean().sqrt()+1e-12)).item()
   self.assertAlmostEqual(e,r['primary_value'],places=8)
if __name__=='__main__':unittest.main()
