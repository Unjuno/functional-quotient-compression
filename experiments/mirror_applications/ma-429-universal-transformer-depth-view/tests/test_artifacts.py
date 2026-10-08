import hashlib,json,pathlib,statistics,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestMA429(unittest.TestCase):
 def test_payloads_and_norms(self):
  art=ROOT/'artifacts';rows=[json.loads(x) for x in (art/'runs.jsonl').read_text().splitlines()];self.assertEqual(len(rows),84)
  for r in rows:
   b=(art/'payloads'/r['payload_file']).read_bytes();self.assertEqual(len(b),r['serialized_bytes']);h=hashlib.sha256(b).hexdigest();self.assertEqual(h,r['payload_sha256']);self.assertEqual(h,r['roundtrip_sha256']);self.assertLessEqual(max(r['spectral_norms_by_depth']),.95001)
 def test_depth_extrapolation_and_bytes(self):
  rows=[json.loads(x) for x in (ROOT/'artifacts/runs.jsonl').read_text().splitlines() if '"split": "fresh"' in x]
  mean=lambda m,k:statistics.mean(r[k] for r in rows if r['method']==m)
  self.assertLess(mean('mirror_time','mean_error_depth5_8'),mean('linear_time','mean_error_depth5_8'))
  self.assertGreater(mean('mirror_time','serialized_bytes'),.9*mean('linear_time','serialized_bytes'))
if __name__=='__main__':unittest.main()
