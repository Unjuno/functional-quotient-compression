import hashlib,json,pathlib,statistics,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestMA425(unittest.TestCase):
 def test_payloads_and_fixed_nfe(self):
  art=ROOT/'artifacts';rows=[json.loads(x) for x in (art/'runs.jsonl').read_text().splitlines()];self.assertEqual(len(rows),84)
  for r in rows:
   b=(art/'payloads'/r['payload_file']).read_bytes();self.assertEqual(len(b),r['serialized_bytes']);h=hashlib.sha256(b).hexdigest();self.assertEqual(h,r['payload_sha256']);self.assertEqual(h,r['roundtrip_sha256'])
  self.assertEqual({r['solver_nfe_rk4_16'] for r in rows},{64})
 def test_t2_quality_byte_gate(self):
  rows=[json.loads(x) for x in (ROOT/'artifacts/runs.jsonl').read_text().splitlines() if '"split": "fresh"' in x]
  mean=lambda m,k:statistics.mean(r[k] for r in rows if r['method']==m)
  self.assertLess(mean('mirror_time','endpoint_nrmse_h2'),1.1*mean('linear_time','endpoint_nrmse_h2'))
  self.assertGreater(mean('mirror_time','serialized_bytes'),.9*mean('linear_time','serialized_bytes'))
if __name__=='__main__':unittest.main()
