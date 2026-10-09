import csv, hashlib, importlib.util, io, json, unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma482_run',ROOT/'source/run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
class ArtifactTests(unittest.TestCase):
 def test_a1_rows_and_payloads_replay(self):
  rows=list(csv.DictReader((ROOT/'RESULTS_CORE.csv').open()))
  self.assertEqual(len(rows),432)
  self.assertEqual({int(r['world']) for r in rows},{48220,48221,48222})
  self.assertEqual({r['method'] for r in rows},{'dense','native_rvq','generic_coeff_rvq','mirror_rvq'})
  for row in rows:
   blob=(ROOT/row['path'].split('ma-482-residual-vq-mirror/',1)[1]).read_bytes()
   self.assertEqual(len(blob),int(row['payload_bytes']))
   self.assertEqual(hashlib.sha256(blob).hexdigest(),row['hash'])
   obj=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
   y=run.make_world(int(row['world']),int(row['seed']))[1]
   err,abl,used=run.evaluate(y,obj)
   self.assertAlmostEqual(err,float(row['normalized_rmse']),places=7)
   self.assertEqual(abl,json.loads(row['stage_ablation_rmse']))
   self.assertEqual(used,json.loads(row['used_codes_per_stage']))
 def test_phase_codes_are_inferred_from_observed_vectors(self):
  _,y=run.make_world(48220,0)
  k,r=16,4
  expected=[]
  for j in range(r):
   phase=torch.atan2(y[:,2*j+1],y[:,2*j])
   expected.append(torch.remainder(torch.round(phase*k/(2*torch.pi)).long(),k).to(torch.uint8))
  self.assertTrue(torch.equal(run.mirror_codes(y,k,r),torch.stack(expected,1)))
if __name__=='__main__':unittest.main()
