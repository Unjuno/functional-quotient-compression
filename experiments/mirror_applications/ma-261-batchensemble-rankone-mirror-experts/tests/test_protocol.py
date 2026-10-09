import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'));import run
class Checks(unittest.TestCase):
 def test_teacher_and_data_shapes(self):
  b,u,v,a=run.teacher(26100,0);(x,y,t),(xt,yt,tt)=run.data(26100,0,b,u,v,a);self.assertEqual(tuple(x.shape),(2048,2));self.assertEqual(tuple(t.shape),(2048,2));self.assertEqual(tuple(xt.shape),(1024,2))
 def test_rank_one_target_family(self):
  b,u,v,a=run.teacher(26100,0);W=b[None]+a[:,None,None]*run.torch.outer(u,v)[None];self.assertEqual(tuple(W.shape),(4,2,2));self.assertTrue(all(int(run.torch.linalg.matrix_rank(z-b))<=1 for z in W))
 def test_dev_grid_and_payloads(self):
  import csv,hashlib
  rows=list(csv.DictReader((ROOT/'DEVELOPMENT_RESULTS.csv').open()));self.assertEqual(len(rows),30)
  for r in rows:
   b=(ROOT.parents[2]/r['path']).read_bytes();self.assertEqual(len(b),int(r['payload_bytes']));self.assertEqual(hashlib.sha256(b).hexdigest(),r['sha256'])
 def test_fresh_grid_payload_hashes_and_mirror_generic_bytes(self):
  import csv,hashlib
  rows=list(csv.DictReader((ROOT/'RESULTS_CORE.csv').open()));self.assertEqual(len(rows),45)
  for r in rows:
   blob=(ROOT.parents[2]/r['path']).read_bytes();self.assertEqual(len(blob),int(r['payload_bytes']));self.assertEqual(hashlib.sha256(blob).hexdigest(),r['sha256'])
  m=[r for r in rows if r['method']=='mirror'];g=[r for r in rows if r['method']=='generic_rank1'];self.assertEqual([r['payload_bytes'] for r in m],[r['payload_bytes'] for r in g])
 def test_fresh_locked(self):
  p=json.loads((ROOT/'PROTOCOL.json').read_text());self.assertEqual(p['fresh']['worlds'],[26110,26111,26112])
if __name__=='__main__':unittest.main()
