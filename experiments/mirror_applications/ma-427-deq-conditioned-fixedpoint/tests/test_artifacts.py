import hashlib,json,pathlib,statistics,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestMA427(unittest.TestCase):
 def test_payloads_and_solver(self):
  art=ROOT/'artifacts';rows=[json.loads(x) for x in (art/'runs.jsonl').read_text().splitlines()];self.assertEqual(len(rows),105)
  for r in rows:
   b=(art/'payloads'/r['payload_file']).read_bytes();self.assertEqual(len(b),r['serialized_bytes']);h=hashlib.sha256(b).hexdigest();self.assertEqual(h,r['payload_sha256']);self.assertEqual(h,r['roundtrip_sha256'])
  fresh=[r for r in rows if r['split']=='fresh'];self.assertEqual(sum(sum(r['nonconvergence_per_code']) for r in fresh),0);self.assertLessEqual(max(r['max_operator_spectral_norm'] for r in fresh),.80001)
 def test_registered_quality_and_bytes_gates_miss(self):
  rows=[json.loads(x) for x in (ROOT/'artifacts/runs.jsonl').read_text().splitlines() if '"split": "fresh"' in x]
  mean=lambda m,k:statistics.mean(r[k] for r in rows if r['method']==m)
  self.assertGreater(mean('mirror','mean_equilibrium_nrmse'),1.1*mean('independent','mean_equilibrium_nrmse'))
  self.assertGreater(mean('mirror','serialized_bytes'),.6*mean('independent','serialized_bytes'))
if __name__=='__main__':unittest.main()
