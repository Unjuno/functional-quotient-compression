import csv,hashlib,io,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
class TestMA519(unittest.TestCase):
 def test_fresh_rows_and_payloads(self):
  with (ROOT/'RESULTS_CORE.csv').open(newline='') as f: rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),45)
  self.assertEqual({int(r['world']) for r in rows},{51910,51911,51912})
  self.assertEqual({int(r['seed']) for r in rows},{0,1,2})
  self.assertEqual({r['method'] for r in rows},{'direct_icl','query_only','oracle_fv','routed_fv','explicit_id_fv_payload'})
  for r in rows:
   if not r['path']: continue
   b=(ROOT/r['path'].split('ma-519-function-vector-context-routing/',1)[1]).read_bytes()
   self.assertEqual(len(b),int(r['payload_bytes']))
   self.assertEqual(hashlib.sha256(b).hexdigest(),r['hash'])
   obj=torch.load(io.BytesIO(b),map_location='cpu',weights_only=False)
   self.assertIn('fv_bank',obj)
 def test_route_and_execution_separate(self):
  with (ROOT/'RESULTS_CORE.csv').open(newline='') as f: rows=list(csv.DictReader(f))
  routed=[r for r in rows if r['method']=='routed_fv'];oracle=[r for r in rows if r['method']=='oracle_fv']
  self.assertEqual(len(routed),9);self.assertEqual(len(oracle),9)
  self.assertTrue(all(float(r['accuracy'])==0 for r in routed+oracle))
  self.assertTrue(any(float(r['routing_accuracy'])<.95 for r in routed))
if __name__=='__main__':unittest.main()
