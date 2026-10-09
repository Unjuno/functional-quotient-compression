import csv,hashlib,io,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
class TestMA520(unittest.TestCase):
 def test_rows_split_and_payloads(self):
  with (ROOT/'RESULTS_CORE.csv').open(newline='') as f: rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),45);self.assertEqual({int(r['world']) for r in rows},{52010,52011,52012});self.assertEqual({int(r['seed']) for r in rows},{0,1,2})
  self.assertEqual({r['method'] for r in rows},{'direct_icl','query_only','explicit','generic_pca','mirror_pq'})
  for r in rows:
   if not r['path']: continue
   b=(ROOT/r['path'].split('ma-520-function-vector-mirror-distillation/',1)[1]).read_bytes()
   self.assertEqual(len(b),int(r['payload_bytes']));self.assertEqual(hashlib.sha256(b).hexdigest(),r['hash'])
   obj=torch.load(io.BytesIO(b),map_location='cpu',weights_only=False);self.assertTrue(obj)
   if r['method']=='generic_pca': self.assertTrue({'mean','basis','coords'}<=set(obj))
   if r['method']=='mirror_pq': self.assertTrue({'centers','ids'}<=set(obj))
   if r['method']=='explicit': self.assertTrue({'vectors','task_labels'}<=set(obj))
 def test_quality_gate(self):
  with (ROOT/'RESULTS_CORE.csv').open(newline='') as f: rows=list(csv.DictReader(f))
  for m in ('direct_icl','explicit','generic_pca','mirror_pq'):
   self.assertEqual(len([r for r in rows if r['method']==m]),9)
  self.assertTrue(all(float(r['accuracy'])==0 for r in rows if r['method'] in ('explicit','generic_pca','mirror_pq')))
if __name__=='__main__':unittest.main()
