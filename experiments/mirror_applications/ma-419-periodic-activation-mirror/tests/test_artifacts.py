import hashlib,json,pathlib,statistics,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestMA419(unittest.TestCase):
 def test_payloads(self):
  art=ROOT/'artifacts';rows=[json.loads(x) for x in (art/'runs.jsonl').read_text().splitlines()];self.assertEqual(len(rows),105)
  for r in rows:
   b=(art/'payloads'/r['payload_file']).read_bytes();self.assertEqual(len(b),r['serialized_bytes']);h=hashlib.sha256(b).hexdigest();self.assertEqual(h,r['payload_sha256']);self.assertEqual(h,r['roundtrip_sha256'])
 def test_per_world_gate_and_latent_control(self):
  rows=[json.loads(x) for x in (ROOT/'artifacts/runs.jsonl').read_text().splitlines() if '"split": "fresh"' in x]
  for w in (41910,41911,41912):
   mean=lambda m:statistics.mean(r['mean_normalized_rmse'] for r in rows if r['world']==w and r['method']==m)
   self.assertEqual(mean('mirror3')>1.1*mean('modnet'),w in (41910,41911))
  self.assertGreater(statistics.mean(r['mean_normalized_rmse'] for r in rows if r['method']=='mirror3'),statistics.mean(r['mean_normalized_rmse'] for r in rows if r['method']=='latent3'))
if __name__=='__main__':unittest.main()
