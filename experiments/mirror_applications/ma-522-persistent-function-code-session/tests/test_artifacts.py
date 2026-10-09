import csv,hashlib,json,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
class TestMA522(unittest.TestCase):
 def test_rows_state_and_context_serialization(self):
  with (ROOT/'RESULTS_CORE.csv').open(newline='') as f: rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),45)
  self.assertEqual({int(r['world']) for r in rows},{52210,52211,52212})
  self.assertEqual({int(r['seed']) for r in rows},{0,1,2})
  self.assertEqual({r['method'] for r in rows},{'repeat_icl','query_only','explicit_session_fv','pca_session_code','pq_session_code'})
  for r in rows:
   self.assertEqual(len(json.loads(r['turn_accuracy_json'])),8)
   self.assertEqual(len(json.loads(r['turn_nll_json'])),8)
   self.assertEqual(int(r['total_state_plus_context_bytes']),int(r['session_state_bytes'])+int(r['cumulative_context_bytes']))
   if r['path']:
    b=(ROOT/r['path'].split('ma-522-persistent-function-code-session/',1)[1]).read_bytes()
    self.assertEqual(len(b),int(r['session_state_bytes']));self.assertEqual(hashlib.sha256(b).hexdigest(),r['hash'])
    obj=torch.load(__import__('io').BytesIO(b),map_location='cpu',weights_only=False);self.assertTrue(obj)
   ctype='icl' if r['method']=='repeat_icl' else 'query'
   c=ROOT/'artifacts'/'contexts'/f"{r['world']}_{r['seed']}_{ctype}_context.pt"
   raw=c.read_bytes();self.assertEqual(len(raw),int(r['cumulative_context_bytes']));self.assertEqual(hashlib.sha256(raw).hexdigest(),r['context_payload_hash'])
 def test_quality_and_turn_retention_gate(self):
  with (ROOT/'RESULTS_CORE.csv').open(newline='') as f: rows=list(csv.DictReader(f))
  means={m:sum(float(r['accuracy']) for r in rows if r['method']==m)/9 for m in {r['method'] for r in rows}}
  self.assertGreater(means['repeat_icl'],.4)
  for m in ('explicit_session_fv','pca_session_code','pq_session_code'):
   self.assertLess(means[m],.05)
if __name__=='__main__':unittest.main()
