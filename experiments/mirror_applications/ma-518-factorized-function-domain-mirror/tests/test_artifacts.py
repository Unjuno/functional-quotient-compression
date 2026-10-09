import csv, hashlib, io, unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
class TestMA518(unittest.TestCase):
 def test_rows_and_payload_hashes(self):
  with (ROOT/'RESULTS_CORE.csv').open(newline='') as f: rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),45)
  self.assertEqual({int(r['world']) for r in rows},{51810,51811,51812})
  self.assertEqual({int(r['seed']) for r in rows},{0,1,2})
  self.assertEqual({r['method'] for r in rows},{'direct_icl','query_only','explicit','generic','factorized'})
  for r in rows:
   if not r['path']: continue
   b=(ROOT/r['path'].split('ma-518-factorized-function-domain-mirror/',1)[1]).read_bytes()
   self.assertEqual(len(b),int(r['payload_bytes']))
   self.assertEqual(hashlib.sha256(b).hexdigest(),r['hash'])
   x=torch.load(io.BytesIO(b),map_location='cpu',weights_only=False)
   self.assertEqual(x['method'],r['method'] if r['method']!='generic' else 'generic_pca')
 def test_quality_gate_failure_replays(self):
  with (ROOT/'RESULTS_CORE.csv').open(newline='') as f: rows=list(csv.DictReader(f))
  for m in ('direct_icl','query_only','explicit','generic','factorized'):
   rr=[r for r in rows if r['method']==m];self.assertEqual(len(rr),9)
  for m in ('explicit','generic','factorized'):
   self.assertTrue(all(float(r['accuracy'])==0 for r in rows if r['method']==m))
  self.assertTrue(all(float(r['accuracy'])>=.3125 for r in rows if r['method']=='direct_icl'))
if __name__=='__main__': unittest.main()
