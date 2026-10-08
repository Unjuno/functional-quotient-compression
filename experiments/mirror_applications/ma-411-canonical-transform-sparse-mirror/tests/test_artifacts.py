import hashlib,json,pathlib,statistics,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestMA411(unittest.TestCase):
 def test_payloads(self):
  art=ROOT/'artifacts';rows=[json.loads(s) for s in (art/'runs.jsonl').read_text().splitlines()];self.assertEqual(len(rows),126)
  for r in rows:
   b=(art/'payloads'/r['payload_file']).read_bytes();self.assertEqual(len(b),r['serialized_bytes']);h=hashlib.sha256(b).hexdigest();self.assertEqual(h,r['payload_sha256']);self.assertEqual(h,r['roundtrip_sha256'])
 def test_fresh_gates(self):
  rows=[json.loads(s) for s in (ROOT/'artifacts/runs.jsonl').read_text().splitlines() if '"split": "fresh"' in s]
  sparse=[r for r in rows if r['method']=='sparse_cond'];dense=[r for r in rows if r['method']=='dense_residual']
  self.assertEqual({r['world'] for r in sparse},{41110,41111,41112});self.assertLessEqual(max(max(r['condition_numbers']) for r in sparse),2)
  self.assertLessEqual(statistics.mean(r['normalized_rmse'] for r in sparse),.1*statistics.mean(r['normalized_rmse'] for r in dense))
  self.assertLess(statistics.mean(r['serialized_bytes'] for r in sparse),.4*statistics.mean(r['serialized_bytes'] for r in dense))
if __name__=='__main__':unittest.main()
