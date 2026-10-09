import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'));import run
class Checks(unittest.TestCase):
 def test_task_shapes(self):
  b,w=run.make_tasks(25800,0,'aligned');self.assertEqual(tuple(w.shape),(8,32,32));self.assertTrue(bool(run.torch.isfinite(w).all()))
 def test_rank2_reconstruction(self):
  b,w=run.make_tasks(25800,0,'random');p=torch_stack([b+run.rank2(b,z)[0]@run.rank2(b,z)[1].T for z in w]);self.assertLess(run.nrmse(p,w),1e-5)
 def test_untied_reference_is_exact(self):
  b,w=run.make_tasks(25800,0,'aligned');_,_,xv,yv=run.data(25800,0,'aligned',w);self.assertLess(run.nrmse(torch_stack([xv[e]@w[e] for e in range(8)]),yv),1e-6)
 def test_metric(self):
  x=run.torch.randn(8,32,32);self.assertLess(run.nrmse(x,x),1e-7)
 def test_dev_rows_and_payload_hashes(self):
  import csv,hashlib
  rows=list(csv.DictReader((ROOT/'DEVELOPMENT_RESULTS.csv').open()));self.assertEqual(len(rows),60)
  for r in rows:
   b=(ROOT.parents[2]/r['path']).read_bytes();self.assertEqual(len(b),int(r['payload_bytes']));self.assertEqual(hashlib.sha256(b).hexdigest(),r['sha256'])
 def test_fresh_locked(self):
  p=json.loads((ROOT/'PROTOCOL.json').read_text());self.assertEqual(p['fresh']['worlds'],[25810,25811,25812])
 def test_fresh_grid_hash_and_aligned_gate(self):
  import csv,hashlib
  rows=list(csv.DictReader((ROOT/'RESULTS_CORE.csv').open()));self.assertEqual(len(rows),90)
  self.assertEqual({int(r['world']) for r in rows},{25810,25811,25812})
  for r in rows:
   blob=(ROOT.parents[2]/r['path']).read_bytes();self.assertEqual(len(blob),int(r['payload_bytes']));self.assertEqual(hashlib.sha256(blob).hexdigest(),r['sha256'])
  for world in [25810,25811,25812]:
   q=[r for r in rows if int(r['world'])==world and r['stratum']=='aligned' and r['method']=='mirror_givens'][0];self.assertLess(float(q['mean_expert_nrmse']),.01)
def torch_stack(xs):return run.torch.stack(xs)
if __name__=='__main__':unittest.main()
