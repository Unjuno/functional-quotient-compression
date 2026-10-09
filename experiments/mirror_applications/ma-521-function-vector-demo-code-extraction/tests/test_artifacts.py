import csv,hashlib,io,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
class T(unittest.TestCase):
 def test_rows_and_hashes(self):
  with (ROOT/'RESULTS_CORE.csv').open(newline='') as f: rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),36);self.assertEqual({int(r['world']) for r in rows},{52110,52111,52112});self.assertEqual({int(r['seed']) for r in rows},{0,1,2})
  self.assertEqual({r['method'] for r in rows},{'direct_icl','query_only','oracle_fv','compiled_mirror'})
  for r in rows:
   if not r['path']:continue
   b=(ROOT/r['path'].split('ma-521-function-vector-demo-code-extraction/',1)[1]).read_bytes();self.assertEqual(len(b),int(r['payload_bytes']));self.assertEqual(hashlib.sha256(b).hexdigest(),r['hash']);self.assertIn('fv_bank',torch.load(io.BytesIO(b),map_location='cpu',weights_only=False))
 def test_task_quality_fails(self):
  with (ROOT/'RESULTS_CORE.csv').open(newline='') as f:rows=list(csv.DictReader(f))
  self.assertTrue(all(float(r['accuracy'])==0 for r in rows if r['method'] in ('oracle_fv','compiled_mirror')))
  self.assertTrue(all(float(r['code_accuracy'])==.25 for r in rows if r['method']=='compiled_mirror'))
if __name__=='__main__':unittest.main()
