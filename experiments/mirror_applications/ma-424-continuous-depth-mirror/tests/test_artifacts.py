import hashlib,json,pathlib,statistics,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestMA424(unittest.TestCase):
 def test_payload_hashes_and_nfe(self):
  art=ROOT/'artifacts';rows=[json.loads(x) for x in (art/'runs.jsonl').read_text().splitlines()];self.assertEqual(len(rows),105)
  for r in rows:
   b=(art/'payloads'/r['payload_file']).read_bytes();self.assertEqual(len(b),r['serialized_bytes']);h=hashlib.sha256(b).hexdigest();self.assertEqual(h,r['payload_sha256']);self.assertEqual(h,r['roundtrip_sha256'])
  self.assertEqual({r['nfe_rk4_16'] for r in rows},{64})
 def test_strict_storage_and_stiffness_gates(self):
  rows=[json.loads(x) for x in (ROOT/'artifacts/runs.jsonl').read_text().splitlines() if '"split": "fresh"' in x]
  mean=lambda m,k:statistics.mean(r[k] for r in rows if r['method']==m)
  self.assertGreater(mean('mirror','serialized_bytes'),.6*mean('independent','serialized_bytes'))
  self.assertGreater(max(max(r['stiffness_ratios']) for r in rows if r['method']=='mirror'),max(max(r['stiffness_ratios']) for r in rows if r['method']=='independent'))
if __name__=='__main__':unittest.main()
