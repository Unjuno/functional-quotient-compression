import hashlib,json,pathlib,statistics,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestMA418(unittest.TestCase):
 def test_payload_hashes(self):
  art=ROOT/'artifacts';rows=[json.loads(x) for x in (art/'runs.jsonl').read_text().splitlines()];self.assertEqual(len(rows),126)
  for r in rows:
   b=(art/'payloads'/r['payload_file']).read_bytes();self.assertEqual(len(b),r['serialized_bytes']);h=hashlib.sha256(b).hexdigest();self.assertEqual(h,r['payload_sha256']);self.assertEqual(h,r['roundtrip_sha256'])
 def test_unseen_pair_gate(self):
  rows=[json.loads(x) for x in (ROOT/'artifacts/runs.jsonl').read_text().splitlines() if '"split": "fresh"' in x]
  mean=lambda m:statistics.mean(r['heldout_pair_normalized_rmse'] for r in rows if r['method']==m)
  self.assertGreater(mean('mirror_factor'),1.1*mean('latent_factor'))
  self.assertEqual({r['world'] for r in rows if r['method']=='mirror_factor'},{41810,41811,41812})
if __name__=='__main__':unittest.main()
