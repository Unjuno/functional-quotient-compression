import hashlib,json,pathlib,statistics,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestMA436(unittest.TestCase):
 def test_payloads_and_steps(self):
  art=ROOT/'artifacts';rows=[json.loads(x) for x in (art/'runs.jsonl').read_text().splitlines()];self.assertEqual(len(rows),84)
  for r in rows:
   b=(art/'payloads'/r['payload_file']).read_bytes();self.assertEqual(len(b),r['serialized_bytes']);h=hashlib.sha256(b).hexdigest();self.assertEqual(h,r['payload_sha256']);self.assertEqual(h,r['roundtrip_sha256'])
  self.assertEqual({r['sequence_length'] for r in rows},{32})
 def test_scoped_pareto_and_compute_gate(self):
  rows=[json.loads(x) for x in (ROOT/'artifacts/runs.jsonl').read_text().splitlines() if '"split": "fresh"' in x]
  mean=lambda m,k:statistics.mean(r[k] for r in rows if r['method']==m)
  self.assertLess(mean('mirror','sequence_nrmse'),mean('independent','sequence_nrmse'))
  self.assertLess(mean('mirror','serialized_bytes'),mean('independent','serialized_bytes'))
  self.assertGreater(mean('mirror','active_mac_proxy_per_token'),mean('independent','active_mac_proxy_per_token'))
  self.assertGreater(mean('mirror','inference_wall_time_per_batch_s'),1.5*mean('independent','inference_wall_time_per_batch_s'))
if __name__=='__main__':unittest.main()
