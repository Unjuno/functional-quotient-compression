import hashlib,json,pathlib,statistics,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestMA416(unittest.TestCase):
 def test_payload_hashes(self):
  art=ROOT/'artifacts';rows=[json.loads(x) for x in (art/'runs.jsonl').read_text().splitlines()];self.assertEqual(len(rows),126)
  for r in rows:
   b=(art/'payloads'/r['payload_file']).read_bytes();self.assertEqual(len(b),r['serialized_bytes']);h=hashlib.sha256(b).hexdigest();self.assertEqual(h,r['payload_sha256']);self.assertEqual(h,r['roundtrip_sha256'])
 def test_fresh_quality_gate(self):
  rows=[json.loads(x) for x in (ROOT/'artifacts/runs.jsonl').read_text().splitlines() if '"split": "fresh"' in x]
  vals={m:statistics.mean(r['mean_normalized_rmse'] for r in rows if r['method']==m) for m in ('mirror2','latent2')}
  self.assertGreater(vals['mirror2'],1.1*vals['latent2'])
if __name__=='__main__':unittest.main()
