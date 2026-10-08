import hashlib,json,pathlib,statistics,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestMA417(unittest.TestCase):
 def test_payload_hashes(self):
  art=ROOT/'artifacts';rows=[json.loads(x) for x in (art/'runs.jsonl').read_text().splitlines()];self.assertEqual(len(rows),189)
  for r in rows:
   b=(art/'payloads'/r['payload_file']).read_bytes();self.assertEqual(len(b),r['serialized_bytes']);h=hashlib.sha256(b).hexdigest();self.assertEqual(h,r['payload_sha256']);self.assertEqual(h,r['roundtrip_sha256'])
 def test_useful_multiplicity_and_quality_gate(self):
  rows=[json.loads(x) for x in (ROOT/'artifacts/runs.jsonl').read_text().splitlines() if '"split": "fresh"' in x]
  def mean(m,k):return statistics.mean(r[k] for r in rows if r['method']==m)
  self.assertGreater(mean('mirror_cb16','mean_normalized_rmse'),1.1*mean('continuous2','mean_normalized_rmse'))
  self.assertLess(mean('mirror_cb16','unique_codes_used'),16)
  for k in (4,8,16):self.assertLess(mean(f'latent_cb{k}','mean_normalized_rmse'),mean(f'mirror_cb{k}','mean_normalized_rmse'))
if __name__=='__main__':unittest.main()
