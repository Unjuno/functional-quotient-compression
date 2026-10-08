import hashlib,json,pathlib,statistics,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestMA434(unittest.TestCase):
 def test_payloads_and_fixed_steps(self):
  art=ROOT/'artifacts';rows=[json.loads(x) for x in (art/'runs.jsonl').read_text().splitlines()];self.assertEqual(len(rows),84)
  for r in rows:
   b=(art/'payloads'/r['payload_file']).read_bytes();self.assertEqual(len(b),r['serialized_bytes']);h=hashlib.sha256(b).hexdigest();self.assertEqual(h,r['payload_sha256']);self.assertEqual(h,r['roundtrip_sha256'])
  self.assertEqual({r['sequence_steps'] for r in rows},{16})
 def test_storage_gate_miss_and_quality_gain(self):
  rows=[json.loads(x) for x in (ROOT/'artifacts/runs.jsonl').read_text().splitlines() if '"split": "fresh"' in x]
  mean=lambda m,k:statistics.mean(r[k] for r in rows if r['method']==m)
  self.assertGreater(mean('mirror_role','serialized_bytes'),.6*mean('independent','serialized_bytes'))
  self.assertLess(mean('mirror_role','mean_sequence_nrmse'),mean('independent','mean_sequence_nrmse'))
if __name__=='__main__':unittest.main()
